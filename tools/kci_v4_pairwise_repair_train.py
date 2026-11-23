"""Repair V4 fixed-action shortcut with ROOT-GROUPED supervised updates.

No new labels, no cross-state DPO, no CCE, no DPR and NO dev/test
optimizer inputs. Each optimizer step includes the verified risky/safe/
authorized states of exactly one independent training root.

Two PREDECLARED diagnostics with identical optimizer updates and groups:
  grouped_ce: native-gold CE, risky 1/2 safe 1/4 authorized 1/4.
  margin_ce: identical CE plus absolute semantic RAW-VERIFY margin penalty.
This is supervised multiclass training, not a cross-history preference pair.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import argparse,json,random,sys,time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_v4_semantic_action import scoring,canonical_v4_state_text,FORMAT

RAW,VERIFY="SEND_RAW_R","VERIFY_PRIVATE_R"
WEIGHTS={"risky":0.5,"safe":0.25,"authorized":0.25}
MARGIN_TARGET=1.0
MARGIN_LOSS_WEIGHT=0.5

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf8") if line.strip()]

def group_from_verified_states(path):
    x=read(path)
    assert len(x)==9 and len({r["comparison_id"] for r in x})==9
    assert sorted(set(r["root"] for r in x))==["DG31","DG32","DG33"]
    groups=defaultdict(dict)
    for r in x:
        assert r["source_split"]=="train" and r["stage"]=="release"
        assert r["teacher_type"]=="V4_real_history_native_teacher"
        assert r["canonical_actions"]==["ABORT","SEND_RAW_R","VERIFY_PRIVATE_R"]
        assert r["true_history_aware_action"]==r["chosen_action"]
        assert r["chosen_action"]=={"risky":VERIFY,"safe":RAW,"authorized":RAW}[r["condition"]]
        assert r["condition"] not in groups[r["root"]]
        groups[r["root"]][r["condition"]]=r
    for root,rows in groups.items():
        assert set(rows)==set(WEIGHTS)
        assert len({r["role"] for r in rows.values()})==1
        assert len({r["stage"] for r in rows.values()})==1
        for kind,row in rows.items():
            assert row["role"] and row["native_state_hash"]
            assert "\nActually received:" in row["prompt"]
            assert "\nInference:" in row["prompt"]
            assert "Candidate action under consideration" not in row["prompt"]
        # Original native source asserts true history-only and authorization
        # controlled pairs; verify text-level invariance independently.
        def nonreceipts(prompt):
            return "\n".join(s for s in canonical_v4_state_text(prompt,rows["safe"]["original_action_order"]).splitlines()
                             if not s.startswith("Actually received:"))
        assert nonreceipts(rows["risky"]["prompt"])==nonreceipts(rows["safe"]["prompt"]),root
        def noauth(prompt):
            return "\n".join(s for s in canonical_v4_state_text(prompt,rows["risky"]["original_action_order"]).splitlines()
                             if not s.startswith("Recipient norm:"))
        assert noauth(rows["risky"]["prompt"])==noauth(rows["authorized"]["prompt"]),root
        assert rows["risky"]["native_state_hash"]==rows["authorized"]["native_state_hash"]
        assert rows["risky"]["native_state_hash"]!=rows["safe"]["native_state_hash"]
    return dict(sorted(groups.items()))

def root_schedule(roots,seed,steps):
    names=sorted(roots)
    rng=random.Random(seed)
    out=[]
    while len(out)<steps:
        order=names[:]
        rng.shuffle(order)
        out.extend(order)
    return out[:steps]

def state_loss(actions,scores,gold,condition,method):
    # Independent same-state CE, never cross-history DPO.
    ix=actions.index(gold)
    ce=-torch.log_softmax(scores.float(),dim=0)[ix]
    raw_verify_delta=scores[actions.index(RAW)].float()-scores[actions.index(VERIFY)].float()
    target=(-1.0 if condition=="risky" else 1.0)
    if method=="grouped_ce":
        margin=torch.zeros_like(ce)
    else:
        # Absolute context-dependent semantic odds requirement,
        # not a preference between actions from DIFFERENT states.
        margin=F.softplus(MARGIN_TARGET-target*raw_verify_delta)
    return ce+MARGIN_LOSS_WEIGHT*margin,ce,margin,raw_verify_delta

def train():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=Path,required=True)
    ap.add_argument("--base",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--method",choices=("grouped_ce","margin_ce"),required=True)
    ap.add_argument("--steps",type=int,default=96)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--seed",type=int,default=7)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    assert args.steps==96 and args.lr==2e-5 and args.seed==7
    grouped=group_from_verified_states(args.train)
    order=root_schedule(grouped,args.seed,args.steps)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    hf=HFActionPolicy(model_path=args.base,trainable=True,
                      quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
                      lora_rank=8,lora_alpha=16,gradient_checkpointing=True,
                      allow_download=False,temperature=1.0,explore_epsilon=0)
    opt=torch.optim.AdamW(hf.trainable_parameters(),lr=args.lr,weight_decay=0)
    assert "fresh-lora" in hf.policy_id
    curve=[];t=time.monotonic();total_forwards=0
    print("V4_REPAIR_TRAIN_START",args.method,"groups",len(grouped),"states",sum(map(len,grouped.values())),
          "steps",args.steps,"seed",args.seed,flush=True)
    for step,root in enumerate(order):
        hf.model.train()
        opt.zero_grad(set_to_none=True)
        losses=[];correct=0;per_state={}
        for cond in ("risky","safe","authorized"):
            row=grouped[root][cond]
            actions,scores=scoring(hf,row["prompt"],row["original_action_order"])
            assert tuple(actions)==("ABORT","SEND_RAW_R","VERIFY_PRIVATE_R")
            loss,ce,margin,rawdelta=state_loss(actions,scores,row["chosen_action"],cond,args.method)
            assert torch.isfinite(loss)
            (WEIGHTS[cond]*loss).backward()
            total_forwards+=len(actions)
            guess=actions[int(scores.detach().argmax())]
            correct+=int(guess==row["chosen_action"])
            per_state[cond]={"label":row["chosen_action"],"pred":guess,
                "ce":float(ce.detach()),"margin_penalty":float(margin.detach()),
                "RAW_minus_VERIFY":float(rawdelta.detach())}
            losses.append(WEIGHTS[cond]*float(loss.detach()))
        grad=torch.nn.utils.clip_grad_norm_(hf.trainable_parameters(),1.,error_if_nonfinite=True)
        assert torch.isfinite(grad)
        opt.step()
        curve.append({"step":step+1,"root":root,"loss":sum(losses),
                      "pred_correct_before_update":correct,
                      "batch_independent_states":3,
                      "per_state":per_state,"grad_norm_before_clip":float(grad)})
        if step==0 or (step+1)%8==0:
            print("V4_REPAIR_TRAIN",args.method,step+1,
                  "root",root,"meanloss",round(curve[-1]["loss"],5),
                  "batch_correct",correct,"delta_risk",round(per_state["risky"]["RAW_minus_VERIFY"],3),
                  "delta_safe",round(per_state["safe"]["RAW_minus_VERIFY"],3),
                  "grad",round(float(grad),3),flush=True)
    args.output.mkdir(parents=True)
    hf.save_adapter(args.output/"adapter")
    meta={
       "cell":"V4_root_grouped_supervised_objective_fix",
       "method":args.method,"base_model":args.base,"model":"Qwen3-4B",
       "seed":args.seed,"lr":args.lr,"steps":args.steps,
       "optimizer":"AdamW","gradient_checkpointing":True,
       "quantization":"nf4_bf16","LoRA_rank":8,"LoRA_alpha":16,
       "LoRA_modules":["q_proj","v_proj"],
       "semantic_action_interface":FORMAT,
       "not_RL":True,"not_cross_history_DPO":True,
       "no_dev_or_final_test_used_in_training":True,
       "training_data_sha256":sha(args.train),
       "n_independent_roots":len(grouped),
       "n_independent_training_states":sum(map(len,grouped.values())),
       "n_original_V4_normal_and_swapped_records":18,
       "no_new_swaps":True,"batch_state_count":3,
       "objective_weights_by_condition":WEIGHTS,
       "semantic_margin_target":MARGIN_TARGET if args.method=="margin_ce" else None,
       "margin_aux_weight":MARGIN_LOSS_WEIGHT if args.method=="margin_ce" else None,
       "per_root_updates":dict(Counter(order)),
       "root_schedule_sha256":sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest(),
       "train_forward_candidate_passes":total_forwards,
       "each_independent_state_revisited":int(args.steps/len(grouped)),
       "training_wall_seconds":round(time.monotonic()-t,2),
       "curve":curve
    }
    (args.output/"TRAINING.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_REPAIR_TRAIN_COMPLETE",args.method,"seconds",meta["training_wall_seconds"],
          "forward_calls",total_forwards,flush=True)
if __name__=="__main__":train()
