"""Paper-compatible single-shared-policy semantic-action DPO.

Same frozen base is reference pi0, one trainable LoRA is policy pi_theta.
Only two VERIFIED feasible actions from same observation form a preference.
No role-specific adapter routing and no hidden native evaluator labels in o.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
from hashlib import sha256
import argparse,json,random,sys,time
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_shared_semantic_action import SharedSemanticPolicy,candidate_prompt
def load(path):
    x=[json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]
    assert all(r["split"]=="train" for r in x)
    return x
def rejected(record):
    action=record["chosen_action"]
    if record["dataset_tag"]=="v31_early_Qstar":
        other={"OPEN_PRIVATE_CHANNEL":"SKIP_PRIVATE_CHANNEL",
               "SKIP_PRIVATE_CHANNEL":"OPEN_PRIVATE_CHANNEL"}[action]
    elif record["dataset_tag"]=="raw_or_private":
        other={"SEND_RAW_R":"VERIFY_PRIVATE_R",
               "VERIFY_PRIVATE_R":"SEND_RAW_R"}[action]
    else:raise ValueError("Only verified paired preference data supported")
    assert action!=other and other in record["observation"]["legal_actions"]
    return other
def margin(scores,actions,record):
    good=record["chosen_action"];bad=rejected(record)
    return scores[actions.index(good)]-scores[actions.index(bad)]
def frozen_base_ref(policy,record):
    ref={}
    obs=record["observation"]
    with torch.inference_mode():
        for act in (record["chosen_action"],rejected(record)):
            lp=policy.reference_log_distribution(
                candidate_prompt(obs,act),("ACCEPT","REJECT"))
            ref[act]=float(lp[0]-lp[1])
    return ref[record["chosen_action"]]-ref[rejected(record)]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--focus",choices=["v31_72","release8","joint_roles"],default="joint_roles")
    p.add_argument("--steps",type=int,default=96)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--beta",type=float,default=2.0)
    p.add_argument("--seed",type=int,default=7)
    args=p.parse_args()
    assert not args.output.exists()
    random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    data={
        "v31_72":load(args.data_root/"V31_TRAIN_72.jsonl"),
        "release8":load(args.data_root/"RELEASE_TRAIN_STRICT8.jsonl"),
    }
    assert len(data["v31_72"])==72 and len(data["release8"])==8
    hf=HFActionPolicy(model_path=args.model,trainable=True,quantization="4bit",
                     dtype="bf16",max_prompt_tokens=4096,lora_rank=8,
                     lora_alpha=16,gradient_checkpointing=True,allow_download=False)
    sem=SharedSemanticPolicy(hf)
    records=[]
    if args.focus=="joint_roles":
        records=[data["release8"][(j//2)%8] if j%2==0 else data["v31_72"][(j//2)%72]
                 for j in range(144)]
    else:records=data[args.focus]
    # Reference cache is evaluated under the frozen base without adapter.
    distinct={(r["observation_sha256"],r["chosen_action"]):r for r in records}
    ref={}
    start=time.monotonic()
    for i,(key,r) in enumerate(distinct.items()):
        ref[key]=frozen_base_ref(hf,r)
    opt=torch.optim.AdamW(hf.trainable_parameters(),lr=args.lr,weight_decay=0.)
    curve=[]
    for step in range(args.steps):
        hf.model.train();opt.zero_grad(set_to_none=True)
        losses=[];correct=0
        for j in range(args.batch):
            row=records[(step*args.batch+j)%len(records)]
            acts,scores=sem.scores(row["observation"])
            d=margin(scores,acts,row)
            delta=d-ref[(row["observation_sha256"],row["chosen_action"])]
            loss=-F.logsigmoid(args.beta*delta)
            (loss/args.batch).backward()
            losses.append(float(loss.detach()))
            correct+=int(d.detach().item()>0)
        grad=torch.nn.utils.clip_grad_norm_(hf.trainable_parameters(),1.0,
                  error_if_nonfinite=True)
        opt.step()
        curve.append({"step":step+1,"loss":sum(losses)/len(losses),
                      "preferred_over_rejected_in_batch":correct,"grad_norm":float(grad)})
        if step==0 or (step+1)%8==0:
            print("SHARED_DPO",args.focus,step+1,round(curve[-1]["loss"],5),
                  correct,"grad",round(float(grad),5),flush=True)
    args.output.mkdir(parents=True)
    hf.save_adapter(args.output/"adapter")
    manifest={
        "architecture":"one_shared_base_one_shared_trainable_adapter",
        "role_adapter_routing":False,
        "objective":"DPO_over_semantic_action_scores",
        "reference":"same_frozen_pretrained_base_under_disable_adapter",
        "teacher_not_a_Qpi_trajectory_claim":True,
        "v31_pair_is_exact_native_Qstar":True,
        "release_pair_is_native_CI_and_safe_suffix":True,
        "focus":args.focus,"beta":args.beta,"steps":args.steps,"batch":args.batch,
        "lr":args.lr,"seed":args.seed,
        "reference_cache_entries":len(ref),
        "runtime_seconds":round(time.monotonic()-start,2),
        "training_curve":curve}
    (args.output/"TRAINING.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print("SHARED_DPO_COMPLETED",args.focus,flush=True)
if __name__=="__main__":main()
