"""Run local text-only PrivacyLens actions or the original Mistral leakage evaluator."""
import argparse
import gzip
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path


def read_jsonl(path, opener=open):
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def prepare_judgments(rows, actions, official):
    if [a["id"] for a in actions] != [row["id"] for row in rows]:
        raise ValueError("Action rows and canonical benchmark IDs must match in full and in order")
    prompts, keys, judgments = [], [], []
    for i, (row, result) in enumerate(zip(rows, actions, strict=True)):
        processed = official.post_process(result["response_raw"].strip())
        action = processed.partition("Action:")[2].strip()
        has_action = "Action:" in processed
        judgments.append(dict(id=row["id"], parsed_action=has_action, action=action, leak_info=False, items=[]))
        for secret in row["secrets"]:
            item = dict(secret=secret, answer=None, leak=False)
            judgments[i]["items"].append(item)
            if has_action:
                prompts.append(official.prepare_prompt_to_identify_leakage(row["user_name"], secret, action))
                keys.append((i, item))
    return prompts, keys, judgments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "judge"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, help="Local actor or Mistral-7B-Instruct-v0.2 directory")
    parser.add_argument("--adapter", help="Optional actor LoRA directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    parser.add_argument("--context", type=int, default=8192)
    args = parser.parse_args()
    if args.stage == "judge" and args.adapter:
        parser.error("Judge accepts only the standalone Mistral checkpoint")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seed list contains a duplicate")
    cases = read_jsonl(args.data / "cases.jsonl.gz", gzip.open)
    if len(cases) != len({c["id"] for c in cases}) or len(cases) != 493:
        raise ValueError("Expected 493 unique benchmark cases")
    args.out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.stage == "generate":
        from evaluate import make_generator
        args.engine, args.max_tokens = "vllm", 2048
        prompts = [tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True) for c in cases]
        if max(len(t) for t in tok(prompts, add_special_tokens=False)["input_ids"]) + args.max_tokens > args.context:
            raise ValueError("Increase --context to accommodate the longest prompt plus the full output budget")
        with ExitStack() as stack:
            generate = make_generator(args, stack)
            for seed in args.seeds:
                outputs = generate(prompts, seed)
                rows = (dict(id=case["id"], seed=seed, model=args.model, adapter=args.adapter, response_raw=s,
                             completion_tokens=n, finish_reason=f) for case, (s, n, f) in zip(cases, outputs, strict=True))
                write_jsonl(args.out / f"actions_{seed}.jsonl", rows)
        return

    from vllm import LLM, SamplingParams
    spec = importlib.util.spec_from_file_location("official_lr", args.data / "official_lr.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.context, seed=0)
    sampling = SamplingParams(temperature=0, max_tokens=1000, stop=["\n\n---"])
    counts = []
    for seed in args.seeds:
        actions = read_jsonl(args.out / f"actions_{seed}.jsonl")
        prompts, keys, judgments = prepare_judgments(cases, actions, official)
        chats = [tok.apply_chat_template([dict(role="user", content=p)], tokenize=False) for p in prompts]
        outputs = llm.generate(chats, sampling) if chats else []
        for (i, item), output in zip(keys, outputs, strict=True):
            item["answer"] = output.outputs[0].text
            item["leak"] = official.parse_leakage_judgment(item["answer"])
            judgments[i]["leak_info"] |= item["leak"]
        write_jsonl(args.out / f"judgments_{seed}.jsonl", judgments)
        counts.append(dict(seed=seed, leaked=sum(r["leak_info"] for r in judgments), cases=len(judgments),
                           unparsed=sum(not r["parsed_action"] for r in judgments)))
    total = sum(row["cases"] for row in counts)
    summary = dict(judge=args.model, method="PrivacyLens Mistral LR with reconstructed SelfCI actor prompt",
                   per_seed=counts, total=total, lr=sum(row["leaked"] for row in counts) / total)
    (args.out / "lr.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
