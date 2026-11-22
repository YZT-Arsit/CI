"""Role-local action-token DPO for KCI pretrained LMs.

Input: TRAIN-only, real native counterfactual preference JSONL with two
currently legal actions and the original actor-visible language prompt.
A frozen pretrained HF causal LM plus fresh LoRA adapter learns only that
acting role. This script NEVER changes native CI verifiers or the evidence.

Portable CPU self-test checks nonzero differentiable updates without GPU.
Actual Qwen3/4B LoRA requires a local HF checkpoint and CUDA;
Ollama GGUF is inference-only, not a trainable HF checkpoint.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,random,sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))

def validate_pairs(path,role):
    allowed={"RecordsAgent","Coordinator","DirectoryAgent","OrderAgent",
             "ComplianceAgent","SourceAgent"}
    if role not in allowed:raise ValueError("Unknown training role")
    records=[json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s.strip()]
    if not records:raise ValueError("No training pairs")
    for r in records:
        assert r["split"]=="train",("Training data from nontrain split",r.get("world"))
        assert r["role"]==role,("Wrong role",r.get("role"))
        assert r.get("state_hash") and r.get("observation_hash")
        actions=r["all_executable_actions"]
        assert isinstance(actions,list) and len(actions)>=2
        assert r["chosen_action"] in actions and r["rejected_action"] in actions
        assert r["chosen_action"]!=r["rejected_action"]
        for label,key in ((r["chosen_letter"],r["chosen_action"]),
                          (r["rejected_letter"],r["rejected_action"])):
            assert len(label)==1 and "A"<=label<="Z"
            assert actions[ord(label)-ord("A")]==key
        assert len(r["prompt"])>100
        assert "native_state" not in r["prompt"] and "oracle_Qstar" not in r["prompt"]
        assert r.get("native_evidence") and len(r["native_evidence"])==2
        if not r.get("not_Qstar_oracle_teacher",False):
            raise ValueError("Requires actual frozen-policy evidence")
    return records

def dpo_one(policy,record,beta):
    actions=tuple(record["all_executable_actions"])
    w=actions.index(record["chosen_action"])
    l=actions.index(record["rejected_action"])
    with torch.no_grad():
        ref=policy.reference_log_distribution(record["prompt"],actions)
        ref_delta=(ref[w]-ref[l]).detach()
    lp=policy.log_distribution(record["prompt"],actions)
    margin=(lp[w]-lp[l])-ref_delta
    loss=-F.logsigmoid(beta*margin)
    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite preference loss")
    return loss,margin,lp[w]-lp[l]

def optimizer_step(policy,pairs,optimizer,beta,grad_clip):
    optimizer.zero_grad(set_to_none=True)
    logs=[]
    for rec in pairs:
        loss,margin,delta=dpo_one(policy,rec,beta)
        (loss/len(pairs)).backward()
        logs.append((float(loss.detach()),float(margin.detach()),float(delta.detach())))
    norm=torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(),
                                         grad_clip,error_if_nonfinite=True)
    optimizer.step()
    return {"mean_loss":sum(v[0] for v in logs)/len(logs),
            "mean_margin":sum(v[1] for v in logs)/len(logs),
            "mean_actor_action_log_odds":sum(v[2] for v in logs)/len(logs),
            "preclip_gradient_norm":float(norm)}

class MockRolePolicy:
    """CPU mathematical smoke, never substitute for a pretrained LLM claim."""
    def __init__(self):
        self.torch=torch
        self.scores=torch.nn.Parameter(torch.zeros(3))
        self.model=torch.nn.Module()
        self.trainable=True
    def log_distribution(self,prompt,actions):
        return torch.log_softmax(self.scores[:len(actions)],dim=0)
    def reference_log_distribution(self,prompt,actions):
        return torch.log_softmax(torch.zeros(len(actions)),dim=0)
    def trainable_parameters(self):return [self.scores]

def self_test(path=None):
    policy=MockRolePolicy()
    if path:
        records=validate_pairs(path,"Coordinator")
        assert all(r["chosen_action"]=="OPEN_PRIVATE_CHANNEL" for r in records),\
            "Sampled CPU self-test currently expects OPEN > SKIP"
    else:
        records=[{"split":"train","role":"Coordinator",
                  "chosen_action":"OPEN_PRIVATE_CHANNEL",
                  "rejected_action":"SKIP_PRIVATE_CHANNEL",
                  "all_executable_actions":["OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL","ABORT"],
                  "chosen_letter":"A","rejected_letter":"B","prompt":"A"*101,
                  "state_hash":"x","observation_hash":"y",
                  "native_evidence":{"OPEN_PRIVATE_CHANNEL":{"safe_success":True},
                                     "SKIP_PRIVATE_CHANNEL":{"safe_success":False}},
                  "not_Qstar_oracle_teacher":True}]
    opt=torch.optim.AdamW(policy.trainable_parameters(),lr=.08,weight_decay=0)
    first=float((policy.scores[0]-policy.scores[1]).detach())
    values=[]
    for i in range(25):
        record=records[i%len(records)]
        values.append(optimizer_step(policy,[record],opt,2.0,1))
    after=float((policy.scores[0]-policy.scores[1]).detach())
    assert after>first+.3
    assert values[-1]["preclip_gradient_norm"]>0
    print("CPU_DPO_GRADIENT_PASS",{"initial_OPEN_minus_SKIP":first,
                "final_OPEN_minus_SKIP":after,"num_updates":25,
                "last_loss":values[-1]["mean_loss"]},flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--pairs",type=Path)
    ap.add_argument("--role",default="Coordinator")
    ap.add_argument("--model-path",type=str)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--steps",type=int,default=80)
    ap.add_argument("--lr",type=float,default=1e-5)
    ap.add_argument("--beta",type=float,default=2.0)
    ap.add_argument("--grad-clip",type=float,default=1.)
    ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--quantize",action="store_true")
    ap.add_argument("--dtype",choices=("fp32","bf16","fp16"),default="bf16")
    ap.add_argument("--max-tokens",type=int,default=4096)
    args=ap.parse_args()
    if args.self_test:
        self_test(args.pairs)
        return
    if not args.pairs or not args.model_path or not args.output:
        ap.error("Actual LLM training requires --pairs --model-path --output")
    if args.output.exists():raise FileExistsError(args.output)
    if args.steps<1 or args.batch_size<1:raise ValueError("Invalid training budget")
    records=validate_pairs(args.pairs,args.role)
    random.seed(args.seed);torch.manual_seed(args.seed)
    from kci.agent_rl.hf_policy import HFActionPolicy
    policy=HFActionPolicy(model_path=args.model_path,trainable=True,
           quantization="4bit" if args.quantize else "none",
           dtype=args.dtype,max_prompt_tokens=args.max_tokens,
           lora_rank=8,lora_alpha=16,gradient_checkpointing=True,
           temperature=1.0,explore_epsilon=0.0,allow_download=False)
    opt=torch.optim.AdamW(policy.trainable_parameters(),
                          lr=args.lr,weight_decay=0)
    logs=[]
    for i in range(args.steps):
        batch=[records[(i*args.batch_size+j)%len(records)]
               for j in range(args.batch_size)]
        policy.model.train()
        step=optimizer_step(policy,batch,opt,args.beta,args.grad_clip)
        logs.append({"step":i+1,**step})
        if i==0 or (i+1)%10==0:
            print("ROLE_DPO",args.role,"step",i+1,
                  "loss",round(step["mean_loss"],5),
                  "gradient",round(step["preclip_gradient_norm"],5),flush=True)
    args.output.mkdir(parents=True)
    policy.save_adapter(args.output/"adapter")
    manifest={"status":"TRAINED_ACTUAL_PRETRAINED_ROLE_ADAPTER",
         "role":args.role,"seed":args.seed,"step_count":args.steps,
         "pair_count":len(records),"beta":args.beta,"lr":args.lr,
         "model_path":args.model_path,
         "source_pairs_sha256":hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
         "only_train_split":True,
         "original_native_truth_unchanged":True,
         "train_curve":logs}
    (args.output/"TRAINING.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
if __name__=="__main__":main()
