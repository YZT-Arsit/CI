"""Train ONE Qwen3-4B LoRA for all KCI roles via order-free semantic actions.

This is direct action CE/teacher-data interface calibration, NOT a new KCI
algorithm result. No oracle/state labels enter any actor prompt.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse,json,random,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_shared_semantic_action import SharedSemanticPolicy,action_pair_loss,canonical_actions
def load(p):
    assert p.is_file(),p
    x=[json.loads(s) for s in p.read_text(encoding="utf8").splitlines() if s.strip()]
    for r in x:
        assert r["split"]=="train"
        assert r["chosen_action"] in canonical_actions(r["observation"])
        assert r["observation"]["legal_actions"]==list(r["observation"]["legal_actions"])
        assert "Qpi" not in r["observation"]
    return x
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--focus",choices=("release8","release8_blind","v31_72","joint_roles","joint_plus_multirole"),required=True)
    p.add_argument("--steps",type=int,default=96)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--seed",type=int,default=7)
    args=p.parse_args()
    assert not args.output.exists()
    assert args.steps>=1 and 1<=args.batch<=8
    data={
        "release8":load(args.data_root/"RELEASE_TRAIN_STRICT8.jsonl"),
        "release8_blind":load(args.data_root/"RELEASE_TRAIN_STRICT8_BLIND.jsonl"),
        "v31_72":load(args.data_root/"V31_TRAIN_72.jsonl"),
        "multirole":load(args.data_root/"MULTIROLE_TRAIN.jsonl"),
    }
    assert len(data["release8"])==8 and len(data["v31_72"])==72
    assert len({x["root"] for x in data["v31_72"]})==12
    # Multi-role demos are ordered by world in the archive. Round-robin
    # actual root+role buckets so short pilots do not consume only early roots.
    from collections import defaultdict,deque
    buckets=defaultdict(deque)
    for row in data["multirole"]:
        buckets[(row["root"],row["role"])].append(row)
    balanced=[]
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:balanced.append(buckets[key].popleft())
    data["multirole"]=balanced
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    hf=HFActionPolicy(model_path=args.model,trainable=True,
        quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
        lora_rank=8,lora_alpha=16,gradient_checkpointing=True,
        allow_download=False,temperature=1.,explore_epsilon=0.)
    model=SharedSemanticPolicy(hf)
    opt=torch.optim.AdamW(hf.trainable_parameters(),lr=args.lr,weight_decay=0.0)
    curve=[];start=time.monotonic();counts=Counter()
    sampled_roots={k:set() for k in data}; sampled_roles={k:set() for k in data}
    def choice(step,j):
        if args.focus in ("release8","release8_blind"):return args.focus
        if args.focus=="v31_72":return "v31_72"
        if args.focus=="joint_roles":return ("release8","v31_72")[(step*args.batch+j)%2]
        return ("release8","v31_72","multirole")[(step*args.batch+j)%3]
    def sample(src,step,j):
        # Deterministic balanced sampling by group; no dev labels.
        rows=data[src]
        i=(step*args.batch+j)//(2 if args.focus=="joint_roles" else
                                3 if args.focus=="joint_plus_multirole" else 1)
        return rows[i%len(rows)]
    print("SHARED_ACTION_START",args.focus,"seed",args.seed,"lr",args.lr,
          "steps",args.steps,"batch",args.batch,
          "base",hf.policy_id,flush=True)
    for step in range(args.steps):
        hf.model.train()
        opt.zero_grad(set_to_none=True)
        current=[];correct=0
        for j in range(args.batch):
            src=choice(step,j);rec=sample(src,step,j)
            actions,scores=model.scores(rec["observation"])
            assert actions==tuple(sorted(actions))
            loss=action_pair_loss(scores,actions,rec["chosen_action"],torch)
            assert torch.isfinite(loss)
            correct+=int(actions[int(scores.argmax())]==rec["chosen_action"])
            (loss/args.batch).backward()
            current.append(float(loss.detach()))
            counts[src]+=1
            sampled_roots[src].add(rec["root"])
            sampled_roles[src].add(rec["role"])
        grad=torch.nn.utils.clip_grad_norm_(hf.trainable_parameters(),1.0,
                                           error_if_nonfinite=True)
        assert torch.isfinite(grad)
        opt.step()
        curve.append({"step":step+1,"loss":sum(current)/len(current),
                      "batch_correct":correct,"batch_n":args.batch,
                      "preclip_grad":float(grad)})
        if step==0 or (step+1)%8==0:
            print("SHARED_ACTION_TRAIN",args.focus,step+1,
                  "loss",round(curve[-1]["loss"],5),"correct",correct,
                  "grad",round(float(grad),5),flush=True)
    args.output.mkdir(parents=True)
    hf.save_adapter(args.output/"adapter")
    meta={
       "train_contract":"single_shared_4B_single_LoRA_all_roles",
       "role_adapter_routing":False,"one_pretrained_base":True,
       "objective":"semantic_action_softmax_CE","architecture":"per_candidate_A_minus_B_yes_no_logit",
       "source_data":"verified_native_actor_observation",
       "v31_Qstar_is_offline_oracle":True,
       "first_release_is_immediate_native_CI_plus_fixed_safe_suffix":True,
       "native_multi_role_demos_not_Qpi":True,
       "not_a_claim_of_ICML_paper_performance":True,
       "focus":args.focus,"seed":args.seed,"steps":args.steps,"batch":args.batch,
       "lr":args.lr,"optimizer":"AdamW","n_per_group":dict(counts),
       "training_roots_by_group":{k:sorted({r["root"] for r in v}) for k,v in data.items()},
       "sampled_roots_by_group":{k:sorted(v) for k,v in sampled_roots.items()},
       "sampled_roles_by_group":{k:sorted(v) for k,v in sampled_roles.items()},
       "train_data_sha256":{k:sha256((args.data_root/{"release8":"RELEASE_TRAIN_STRICT8.jsonl",
                      "release8_blind":"RELEASE_TRAIN_STRICT8_BLIND.jsonl",
                      "v31_72":"V31_TRAIN_72.jsonl","multirole":"MULTIROLE_TRAIN.jsonl"}[k]).read_bytes()).hexdigest() for k in data},
       "base_model":args.model,"elapsed_seconds":round(time.monotonic()-start,2),
       "curve":curve,
    }
    (args.output/"TRAINING.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("SHARED_ACTION_TRAIN_COMPLETE",args.focus,round(time.monotonic()-start,1),flush=True)
if __name__=="__main__":main()
