"""KCI Agent RL: preflight, full rollout probe, bounded LoRA GRPO pilot.

This is a constrained-action language-model RL baseline, not an implementation
of CCE/DPR. All evidence stays in one fresh output folder.
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import UniformPolicy,aggregate,group_signal,prompt_for,sample_group,scenarios
from experiments.receipt_dataset import SOURCE

def hash_file(p):
    h=sha256()
    with p.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()
def model_fingerprint(model_path,adapter):
    roots=[("base",model_path),("adapter",adapter)]
    result={}
    for label,root in roots:
        if not root:continue
        path=Path(root)
        if not path.is_dir():
            result[label]={"sha256_complete":False,"identifier":str(root)}
            continue
        files=[]
        for p in sorted(path.iterdir()):
            if p.is_file() and (p.name.endswith(".safetensors") or
                  p.name in ("config.json","tokenizer.json","tokenizer_config.json",
                             "model.safetensors.index.json","adapter_config.json",
                             "generation_config.json")):
                files.append((p.name,p))
        result[label]={"sha256_complete":bool(files),
                       "files":{name:hash_file(p) for name,p in files}}
    return result
def write(p,data):
    if p.exists():raise FileExistsError(str(p))
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
def append(p,row):
    with p.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
def provenance(source_root):
    from kci.dynamic_tasks import DynamicTask
    hashes={}
    for kind in ("meeting","query"):
        env=DynamicTask(kind,source_root,related=False,allow=False,horizon=11)
        if not env.source_hashes:raise RuntimeError("Source provenance missing")
        for name,value in env.source_hashes.items():
            if name in hashes and hashes[name]!=value:
                raise RuntimeError("Conflicting source hash for "+name)
            hashes[name]=value
    return hashes
def policy(args,trainable=False):
    if args.backend=="random":
        if trainable:raise ValueError("Cannot optimize random policy")
        return UniformPolicy()
    if not args.model:raise ValueError("HF backend requires --model")
    from kci.agent_rl.hf_policy import HFActionPolicy
    return HFActionPolicy(
       model_path=args.model,adapter=args.adapter,trainable=trainable,
       quantization="4bit" if args.bits4 else "none",dtype=args.dtype,
       max_prompt_tokens=args.max_prompt_tokens,
       allow_download=args.allow_download,lora_rank=args.lora_rank,
       lora_alpha=args.lora_alpha,gradient_checkpointing=args.gradient_checkpointing)
def make_fresh(path):
    if path is None:raise ValueError("--output required")
    p=Path(path).resolve()
    if p.exists() and (not p.is_dir() or any(p.iterdir())):raise FileExistsError("Output exists: "+str(p))
    p.mkdir(parents=True,exist_ok=True)
    return p
def world_order(split,seed,limit):
    grid=scenarios(split)
    random.Random(seed).shuffle(grid)
    if not 1<=limit<=len(grid):raise ValueError("Invalid --world-limit")
    return grid[:limit]
def record_group(out,group,meta):
    for ep in group:append(out/"episodes.jsonl",ep.serialize())
    append(out/"groups.jsonl",meta)
def evaluate(policy_obj,out,split,seed,limit,G,source_root):
    all_groups=[]
    for idx,scenario in enumerate(world_order(split,seed,limit)):
        group=sample_group(scenario,policy_obj,G,seed+idx*19997,split=split,source_root=source_root)
        all_groups.append(group)
        record_group(out,group,{
            "group_index":idx,"scenario":scenario.key,"policy_id":policy_obj.policy_id,
            "rewards":[e.reward for e in group],
            "success":sum(e.success and not e.violation for e in group),
            "violation":sum(e.violation for e in group),
            "zero_variance":len({e.reward for e in group})==1,
            "signal_kind":group_signal(group)})
    return aggregate(all_groups)
def train(policy_obj,out,args,source_root):
    from kci.agent_rl.grpo import GRPOConfig,GRPOUpdater
    cfg=GRPOConfig(lr=args.lr,clip_eps=args.clip_eps,kl_coef=args.kl_coef,
                   entropy_coef=args.entropy_coef,grad_clip=args.grad_clip,
                   train_cost_only=args.train_cost_only)
    write(out/"GRPO_CONFIG.json",asdict(cfg))
    updater=GRPOUpdater(policy_obj,cfg)
    dev_seed=args.seed+700000
    before_dir=out/"dev_before";before_dir.mkdir()
    before=evaluate(policy_obj,before_dir,"dev",dev_seed,args.dev_world_limit,args.group_size,source_root)
    write(before_dir/"SUMMARY.json",before)
    worlds=world_order("train",args.seed,len(scenarios("train")))
    for idx in range(args.max_updates):
        world=worlds[idx%len(worlds)]
        policy_obj.policy_id=f"hf:{args.model}:old_step_{updater.optimizer_steps}"
        group=sample_group(world,policy_obj,args.group_size,
                           args.seed+1000000+2897*idx,source_root=source_root)
        result=updater.update(group)
        result["group_index"]=idx
        record_group(out,group,result)
        print(json.dumps({"group":idx+1,"updated":result["updated"],
               "safe_success":result["safe_success"],
               "violations":result["violations"],"optimizer_steps":updater.optimizer_steps}),flush=True)
    policy_obj.save_adapter(out/"adapter")
    policy_obj.policy_id=f"hf:{args.model}:updated_step_{updater.optimizer_steps}"
    after_dir=out/"dev_after";after_dir.mkdir()
    after=evaluate(policy_obj,after_dir,"dev",dev_seed,args.dev_world_limit,args.group_size,source_root)
    write(after_dir/"SUMMARY.json",after)
    write(out/"SUMMARY.json",{"stage":"bounded_on_policy_action_grpo",
        "before":before,"after":after,"groups_attempted":args.max_updates,
        "optimizer_steps":updater.optimizer_steps,
        "same_dev_scenarios_seeds":True,
        "real_language_model_updated":updater.optimizer_steps>0,
        "learning_signal_gate_passed":updater.optimizer_steps>0,
        "kci_method_superiority_proven":False,
        "independent_business_generalization":False,"large_gpu_go":False})
def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("preflight","probe","train"))
    p.add_argument("--backend",choices=("random","hf"),default="random")
    p.add_argument("--model");p.add_argument("--adapter")
    p.add_argument("--allow-download",action="store_true")
    p.add_argument("--output")
    p.add_argument("--source-root",default=str(SOURCE))
    p.add_argument("--split",choices=("train","dev"),default="dev")
    p.add_argument("--group-size",type=int,default=4)
    p.add_argument("--world-limit",type=int,default=8)
    p.add_argument("--dev-world-limit",type=int,default=8)
    p.add_argument("--seed",type=int,default=9111926)
    p.add_argument("--max-updates",type=int,default=8)
    p.add_argument("--dtype",choices=("fp32","fp16","bf16"),default="fp32")
    p.add_argument("--bits4",action="store_true")
    p.add_argument("--max-prompt-tokens",type=int,default=4096)
    p.add_argument("--lora-rank",type=int,default=8)
    p.add_argument("--lora-alpha",type=int,default=16)
    p.add_argument("--gradient-checkpointing",action="store_true")
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--clip-eps",type=float,default=.2)
    p.add_argument("--kl-coef",type=float,default=.02)
    p.add_argument("--entropy-coef",type=float,default=0.)
    p.add_argument("--grad-clip",type=float,default=1.)
    p.add_argument("--train-cost-only",action="store_true")
    p.add_argument("--confirm-gpu-training",action="store_true")
    p.add_argument("--allow-cpu-smoke",action="store_true")
    return p.parse_args()
def main():
    a=parse_args()
    root=Path(a.source_root).resolve()
    sources=provenance(root)
    if a.mode=="preflight":
        w=scenarios("train")[0].build(root);s=w.decision_start()
        import importlib.util
        cuda=False;gpu_count=0
        if importlib.util.find_spec("torch"):
            import torch
            cuda=torch.cuda.is_available();gpu_count=torch.cuda.device_count()
        print(json.dumps({"source_hashes":sources,"preflight":"PASS",
             "first_prompt_chars":len(prompt_for(w.observation(s),tuple(w.options(s)))),
             "n_train_conditions":len(scenarios("train")),
             "n_dev_conditions":len(scenarios("dev")),
             "cuda_available":cuda,"cuda_devices":gpu_count,
             "hf_model_not_loaded":True,"no_training":True}));return
    if not 2<=a.group_size<=32:raise ValueError("Use 2..32 rollouts per group")
    if a.mode=="train":
        if a.backend!="hf" or not a.confirm_gpu_training:
            raise ValueError("Real HF training requires --confirm-gpu-training")
        if a.adapter:raise ValueError("v1 training only from fresh LoRA, not reused adapter")
        if not 1<=a.max_updates<=128:raise ValueError("Training capped at 128 attempted groups")
        import torch
        if not torch.cuda.is_available():
            if not a.allow_cpu_smoke or a.bits4 or a.dtype!="fp32":
                raise RuntimeError("No CUDA GPU; tiny CPU smoke requires --allow-cpu-smoke --dtype fp32")
        if torch.cuda.is_available() and a.bits4 and a.dtype=="bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA GPU cannot run bf16")
    if not a.output:raise ValueError("--output required")
    obj=policy(a,trainable=a.mode=="train")
    out=make_fresh(a.output)
    revision=subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],
                            capture_output=True,text=True,check=False).stdout.strip()
    files={p.name:hash_file(p) for p in (ROOT/"code"/"kci"/"agent_rl").glob("*.py")}
    write(out/"RUN_CONFIG.json",{**vars(a),"source_hashes":sources,
        "git_commit":revision,"code_hashes":files,
        "model_fingerprints":model_fingerprint(a.model,a.adapter),
        "source_adapted_synthetic_not_real_task_benchmark":True})
    if a.mode=="probe":
        res=evaluate(obj,out,a.split,a.seed,a.world_limit,a.group_size,root)
        write(out/"SUMMARY.json",{**res,"model_training_occurred":False,
                                  "policy_backend":a.backend,"model_weights_verified_for_action_tokens":a.backend=="hf"})
    else:
        train(obj,out,a,root)
    manifest={p.relative_to(out).as_posix():hash_file(p) for p in out.rglob("*") if p.is_file()}
    write(out/"MANIFEST.json",manifest)
    print(json.dumps({"finished":True,"stage":a.mode,"output":str(out),
                      "hash_files":len(manifest)}),flush=True)
if __name__=="__main__":main()
