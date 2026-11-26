"""Consume the unified release using full semantic completions and a frozen DPO reference."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

ROUTES=("sft","dpo","task_sft","task_dpo","static_evidence","process_sft")


def execution_role(r):
    ctx=r["actor_context"]
    actor=r["actor"]
    if ctx.get("role_stage") or ctx.get("role"):
        return ctx.get("role_stage") or ctx["role"]
    if actor==ctx.get("original_holder"):return "source_holder"
    if actor==ctx.get("executor"):return "executor"
    if actor in ctx.get("delegates",{}).values():return "delegate"
    for role,name in ctx.get("source_roles",{}).items():
        if role!="f" and actor==name:return role
    if ctx.get("phase")=="original_business_finalize":return "executor"
    return r["role"]


def examples(path,route,tier=None,value_tier=None,role=None):
    if route not in ROUTES:
        raise ValueError(f"Not a training objective: {route}")
    name="train_"+route
    m=json.loads((path/"MANIFEST.json").read_text(encoding="utf8"))
    assert m["format"]=="kci-role-actions-v1"
    file=path/(name+".jsonl.gz")
    if hashlib.sha256(file.read_bytes()).hexdigest()!=m["shards"][name]["sha256"]:
        raise ValueError(f"Shard hash mismatch: {name}")
    count=0
    with gzip.open(file,"rt",encoding="utf8") as f:
        for line in f:
            r=json.loads(line);count+=1
            if r["partition"]!="train" or r["route"]!=route:
                raise ValueError("Invalid training partition or objective")
            if ((tier is None or tier in r["tiers"]) and (value_tier is None or value_tier in r["value_tiers"])
                    and (role is None or execution_role(r)==role)):
                yield r
    assert count==m["shards"][name]["rows"]


def prompt(r):
    if r["route"]=="process_sft":
        instruction='Return JSON with a private_record and an action from the executable actions. Only action is delivered.'
    elif r["executable_actions"]:
        instruction="Return one complete executable action as JSON."
    else:
        instruction="Return JSON satisfying the role's output contract."
    # On both teacher forcing and autonomous generation only these two fields reach the policy.
    visible={k:r[k] for k in ("actor_context","executable_actions")}
    return instruction+"\n"+json.dumps(visible,sort_keys=True,ensure_ascii=False,separators=(",",":"))


def root_order(rows,rng):
    groups=defaultdict(list)
    for r in rows:groups[r["group"]].append(r)
    group_list=list(groups.values())
    for seq in group_list:rng.shuffle(seq)
    while group_list:
        rng.shuffle(group_list)
        for seq in group_list:yield seq.pop()
        group_list=[seq for seq in group_list if seq]


def run(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel

    if not args.smoke and (not args.route or not args.output):
        raise ValueError("Training requires --route and a new --output directory")
    if args.output and args.output.exists():
        raise ValueError("Output already exists")
    if args.epochs<1 or args.beta<=0 or args.lr<=0:
        raise ValueError("Invalid optimization parameters")
    routes=ROUTES if args.smoke else [args.route]
    rows=[]
    rng=random.Random(args.seed)
    for route in routes:
        seq=examples(args.data,route,args.tier,args.value_tier,args.role)
        if args.smoke:
            best=min(seq,key=lambda r:len(prompt(r))+sum(map(len,r["accepted"][:1]+r["rejected"][:1])),default=None)
            seq=[best] if best else []
        else:
            seq=list(seq)
        if not seq:raise ValueError(f"No records for {route} and tier {args.tier}")
        rows.extend(seq)
    torch.manual_seed(args.seed)
    tok=AutoTokenizer.from_pretrained(args.base,local_files_only=True)
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    dtype=torch.bfloat16 if device!="cpu" else torch.float32
    backbone=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=dtype,device_map={"":device},local_files_only=True)
    model=PeftModel.from_pretrained(backbone,args.adapter,adapter_name="policy",is_trainable=True)
    model.load_adapter(args.adapter,adapter_name="reference",is_trainable=False)
    model.set_adapter("policy")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.train()
    # Deterministic reference and policy log-probabilities, including LoRA dropout.
    for layer in model.modules():
        if isinstance(layer,torch.nn.Dropout):layer.p=0.
    params=[v for k,v in model.named_parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.)
    if args.output:args.output.mkdir(parents=True)
    sink=(args.output/"updates.jsonl").open("w",encoding="utf8") if args.output else None
    budget=dict(updates=0,policy_forward=0,reference_forward=0,prompt_tokens=0,completion_tokens=0)
    started=time.monotonic()
    smoke=[]
    def logp(pre,tail,ref=False):
        ids=torch.tensor([pre+tail],device=model.device)
        logits=model(input_ids=ids,use_cache=False,logits_to_keep=len(tail)+1).logits[0,:-1].float()
        truth=ids[0,-len(tail):]
        budget["reference_forward" if ref else "policy_forward"]+=1
        budget["prompt_tokens"]+=len(pre);budget["completion_tokens"]+=len(tail)
        return -F.cross_entropy(logits,truth,reduction="sum")
    try:
        for epoch in range(1 if args.smoke else args.epochs):
            for r in (rows if args.smoke else root_order(rows,rng)):
                pre=tok.apply_chat_template([{"role":"user","content":prompt(r)}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                if hasattr(pre,"input_ids"):pre=pre.input_ids
                if not isinstance(pre,list):pre=pre.tolist()
                offset=int(r["record_id"][:8],16)
                chosen=r["accepted"][(offset+epoch)%len(r["accepted"])]
                encode=lambda text:tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
                plus=encode(chosen)
                neg=encode(r["rejected"][0]) if r["route"].endswith("dpo") else []
                if len(pre)+max(len(plus),len(neg))>model.config.max_position_embeddings:
                    raise ValueError("Context exceeds model window. Truncation is disabled.")
                opt.zero_grad(set_to_none=True)
                if neg:
                    model.set_adapter("reference")
                    with torch.no_grad():base_margin=logp(pre,plus,True)-logp(pre,neg,True)
                    model.set_adapter("policy")
                    loss=-F.logsigmoid(args.beta*(logp(pre,plus)-logp(pre,neg)-base_margin))
                else:
                    loss=-logp(pre,plus)/len(plus)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                if not torch.isfinite(loss) or (args.smoke and norm.item()==0):raise RuntimeError("Invalid loss or gradient")
                opt.step();budget["updates"]+=1
                rec=dict(route=r["route"],row=r["record_id"],group=r["group"],tier=r["tiers"],
                         loss=loss.item(),gradient_norm=norm.item(),epoch=epoch+1,prompt_tokens=len(pre),
                         chosen_tokens=len(plus),rejected_tokens=len(neg))
                if sink:sink.write(json.dumps(rec)+"\n")
                if args.smoke:smoke.append(rec)
    finally:
        if sink:sink.close()
    if args.output and not args.smoke:model.save_pretrained(args.output/"adapter",selected_adapters=["policy"])
    if device!="cpu":torch.cuda.synchronize()
    result=dict(smoke=args.smoke,device=torch.cuda.get_device_name() if device!="cpu" else device,base=str(args.base),starting_adapter=str(args.adapter),
                adapter_sha256=hashlib.sha256((args.adapter/"adapter_model.safetensors").read_bytes()).hexdigest(),
                data_manifest_sha256=hashlib.sha256((args.data/"MANIFEST.json").read_bytes()).hexdigest(),
                budget=budget,wall_seconds=time.monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated() if device!="cpu" else None,smoke_rows=smoke)
    if args.output:(args.output/"run.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--adapter",type=Path,required=True)
    ap.add_argument("--route",choices=ROUTES)
    ap.add_argument("--tier")
    ap.add_argument("--role")
    ap.add_argument("--value-tier",choices=("SUCCESS","PROTECTION_COST"))
    ap.add_argument("--output",type=Path)
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--lr",type=float,default=5e-5)
    ap.add_argument("--beta",type=float,default=0.1)
    ap.add_argument("--smoke",action="store_true")
    run(ap.parse_args())

