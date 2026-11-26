"""Generate native KCI role actions from a label-free packet, then compare two sealed runs."""
import argparse
import gc
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import ExitStack


def read(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f]


def dump(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from importlib.metadata import version

    rows = read(args.prompts)
    if any(set(row) != {"prompt_id", "prompt"} for row in rows):
        raise ValueError("Expected only identifiers and role-visible prompts for inference")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    records = []
    for r in rows:
        ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=False)
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        records.append((r["prompt_id"], ids))
    records.sort(key=lambda p: (len(p[1]), p[0]))
    if max(len(x) for _, x in records) + args.max_tokens > args.context:
        raise ValueError("Increase context; prompt truncation is disabled")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = dict(prompts_sha256=digest(args.prompts), model=os.environ.get("KCI_MODEL_ID", str(args.model.resolve())), greedy=True,
                    adapter_sha256=digest(args.adapter / "adapter_model.safetensors") if args.adapter else None,
                    engine=args.engine, versions={k: version(k) for k in {"torch","transformers",args.engine}},
                    max_tokens=args.max_tokens, context=args.context, batch=args.batch, prompts=len(records))
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf8")) != manifest:
        raise ValueError("Output directory bound to another run")
    manifest_path.write_text(dump(manifest), encoding="utf8")
    out = args.out / "generations.jsonl"
    done = {r["prompt_id"] for r in map(json.loads, out.read_text(encoding="utf8").splitlines())} if out.exists() else set()
    remaining = [(key, ids) for key, ids in records if key not in done]
    if not remaining:
        return
    with ExitStack() as stack:
        model_ref = str(args.model)
        if args.engine == "transformers" or args.adapter:
            device = "cpu" if args.engine == "vllm" else "cuda:0"
            model = AutoModelForCausalLM.from_pretrained(model_ref, dtype=torch.bfloat16, device_map={"": device}, local_files_only=True)
            if args.adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
            model.eval()
        if args.engine == "vllm":
            from vllm import LLM, SamplingParams
            if args.adapter:
                model_ref = stack.enter_context(TemporaryDirectory(dir=os.environ.get("KCI_MERGE_DIR", args.out), prefix="merge-"))
                model.save_pretrained(model_ref)
                tok.save_pretrained(model_ref)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            llm = LLM(model=model_ref, max_model_len=args.context, gpu_memory_utilization=0.85,
                      tensor_parallel_size=1, dtype="bfloat16", seed=42)
            sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=42)
        else:
            eos = model.generation_config.eos_token_id
            stop_ids = set(eos if isinstance(eos, list) else [eos])
        f = stack.enter_context(out.open("a", encoding="utf8"))
        for start in range(0, len(remaining), args.batch):
            batch = remaining[start:start + args.batch]
            if args.engine == "vllm":
                requests = llm.generate([dict(prompt_token_ids=x) for _, x in batch], sampling, use_tqdm=False)
                outputs = [(r.outputs[0].text, len(r.outputs[0].token_ids), r.outputs[0].finish_reason) for r in requests]
            else:
                inputs = tok.pad({"input_ids":[x for _, x in batch]}, padding=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    all_ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_tokens,
                                             pad_token_id=tok.pad_token_id)
                outputs = []
                for tail in all_ids[:, inputs.input_ids.shape[1]:].tolist():
                    stop = next((i for i,t in enumerate(tail) if t in stop_ids), None)
                    if stop is not None:
                        tail = tail[:stop + 1]
                    outputs.append((tok.decode(tail, skip_special_tokens=True), len(tail), "stop" if stop is not None else "length"))
            for (key, ids), (text, n, reason) in zip(batch, outputs, strict=True):
                f.write(dump(dict(prompt_id=key,text=text,prompt_tokens=len(ids),completion_tokens=n,finish_reason=reason))+"\n")
            f.flush()
            print(f"generated {len(done) + min(start + args.batch, len(remaining))}/{len(records)}", flush=True)


def unique_object(pairs):
    d = dict(pairs)
    if len(d) != len(pairs):
        raise ValueError("duplicate JSON key")
    return d


def outcome(r, raw, ctx):
    try:
        obj = json.loads(raw, object_pairs_hook=unique_object)
        pred = dump(obj)
    except (TypeError, ValueError, RecursionError):
        return "invalid_json"
    if not isinstance(obj, dict):
        return "invalid_contract"
    accepted = [json.loads(x) for x in r["accepted"]]
    if r["route"] == "static_evidence":
        ids = obj.get("selected_entry_ids")
        visible = {x["entry_id"] for x in ctx["evidence_available_from_tools"]}
        if (set(obj) != {"selected_entry_ids"} or not isinstance(ids, list) or
                not all(isinstance(x,str) and x in visible for x in ids) or len(set(ids)) != len(ids)):
            return "invalid_contract"
        return "accepted" if any(set(ids)==set(a["selected_entry_ids"]) for a in accepted) else "other_output"
    if r["executable_actions"] and pred not in {dump(a) for a in r["executable_actions"]}:
        return "off_menu"
    if pred in {dump(x) for x in accepted}:
        return "accepted"
    if pred in {dump(json.loads(x)) for x in r["rejected"]}:
        return "rejected"
    if r["executable_actions"]:
        return "outside_closed_set" if "closed_action_set" in r["evidence"] else "legal_unlabelled"
    # Two TOP-R completions have a native three-field typed output contract.
    if not any(set(obj)==set(a) and all(type(obj[k]) is type(a[k]) for k in a) for a in accepted):
        return "invalid_contract"
    return "other_output"


def compare(args):
    labels = read(args.labels)
    prompts = {r["prompt_id"]:r["prompt"] for r in read(args.prompts)}
    assert len(labels)==4133 and len({r["record_id"] for r in labels})==4133 and len({r["group"] for r in labels})==40
    assert {r["prompt_id"] for r in labels} == set(prompts)
    runs = {}
    manifests = {}
    for name, folder in (("base",args.base_run),("selfci",args.selfci_run)):
        manifests[name] = json.loads((folder/"manifest.json").read_text(encoding="utf8"))
        assert manifests[name]["prompts_sha256"]==digest(args.prompts)
        rows = list(map(json.loads,(folder/"generations.jsonl").read_text(encoding="utf8").splitlines()))
        runs[name] = {r["prompt_id"]:r for r in rows}
        assert len(rows)==len(prompts) and set(runs[name])==set(prompts)
    assert manifests["base"]["adapter_sha256"] is None and manifests["selfci"]["adapter_sha256"]
    assert all(manifests["base"][k]==v for k,v in manifests["selfci"].items() if k!="adapter_sha256")
    args.out.mkdir(parents=True,exist_ok=True)
    stats = defaultdict(lambda:defaultdict(Counter))
    with gzip.open(args.out/"paired.jsonl.gz","wt",encoding="utf8") as f:
        for r in labels:
            key=r["prompt_id"]
            ctx=json.loads(prompts[key].split("\n",1)[1])["actor_context"]
            status={name:outcome(r,rs[key]["text"],ctx) for name,rs in runs.items()}
            menu=len(r["executable_actions"])
            slices=["all","group/"+r["group"],"domain/"+r["domain"],"role/"+r["execution_role"],"route/"+r["route"],
                    "domain_role/"+r["domain"]+"/"+r["execution_role"],
                    "menu/"+("forced" if menu==1 else "choice" if menu else "open_contract"),
                    *["evidence/"+e for e in r["evidence"]]]
            for sub in slices:
                for name,s in status.items():stats[sub][name][s]+=1
                stats[sub]["transitions"][status["base"]+"->"+status["selfci"]]+=1
            paired={**r, "predictions":{name:{**rs[key],"status":status[name]} for name,rs in runs.items()}}
            f.write(dump(paired)+"\n")
    groups=[d for s,d in stats.items() if s.startswith("group/")]
    macro={k:sum(g[k]["accepted"]/sum(g[k].values()) for g in groups)/len(groups) for k in runs}
    (args.out/"summary.json").write_text(json.dumps(dict(records=len(labels),groups=len(groups),unique_prompts=len(prompts),
        labels_sha256=digest(args.labels),group_macro_reference_match=macro,slices=stats,manifests=manifests),
        ensure_ascii=False,indent=2),encoding="utf8")
    print(dict(records=len(labels),groups=len(groups),outcomes=dict(stats["all"]),group_macro=macro))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="mode",required=True)
    gen=subs.add_parser("generate")
    gen.add_argument("--model",type=Path,required=True)
    gen.add_argument("--adapter",type=Path)
    gen.add_argument("--engine",choices=["transformers","vllm"],default="vllm")
    gen.add_argument("--max-tokens",type=int,default=384)
    gen.add_argument("--context",type=int,default=8192)
    gen.add_argument("--batch",type=int,default=32)
    comp=subs.add_parser("compare")
    comp.add_argument("--base-run",type=Path,required=True)
    comp.add_argument("--selfci-run",type=Path,required=True)
    comp.add_argument("--labels",type=Path,required=True)
    for command in (gen,comp):
        command.add_argument("--prompts",type=Path,required=True)
        command.add_argument("--out",type=Path,required=True)
    params=parser.parse_args()
    (generate if params.mode=="generate" else compare)(params)
