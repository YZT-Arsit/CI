"""Train Qwen3-4B on V4 native same-state pairs, without touching older trainers.

Supports arbitrary real role/stage/option_order/semantic chosen+rejected.
Uses exactly the original normal+swapped records supplied by V4; a
pair of orientations contributes ONE independent comparison / update.
Only TRAIN split accepted, never use DG36 or finite-native Q* as LM Qpi.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse,json,random,sys,time
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy

ACTION_INSTRUCTION="\nReply with exactly one option letter from the executable options above.\nAnswer:"
FORMAT="v4_literal_native_option_letters_v1"
def model_input(row):
    # Same suffix and same role-visible content for frozen, CE and DPO.
    return row["prompt"]+ACTION_INSTRUCTION
def read_groups(path,release_only):
    x=[json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s.strip()]
    assert x
    seen=set()
    for group in x:
        assert group["group_id"] not in seen
        seen.add(group["group_id"])
        assert group["root"] in ("DG31","DG32","DG33")
        assert group["stage"] in ("release","A_SEND","preflight")
        if release_only:assert group["stage"]=="release"
        assert group["n_original_records"]==len(group["paired_option_records"])==2
        assert len({r["prompt"] for r in group["paired_option_records"]})==2
        assert len({r["role"] for r in group["paired_option_records"]})==1
        assert len({r["stage"] for r in group["paired_option_records"]})==1
        for row in group["paired_option_records"]:
            assert row["role"]==group["role"] and row["stage"]==group["stage"]
            assert row["chosen_action"]==group["chosen_action"]
            assert row["rejected_action"]==group["rejected_action"]
            assert row["source_split"]=="train" and row["not_model_Qpi"]
            actions=row["option_order"]
            assert actions.index(row["chosen_action"])==ord(row["chosen_letter"])-65
            assert actions.index(row["rejected_action"])==ord(row["rejected_letter"])-65
            assert row["evidence"]["grade"]==row["evidence_grade"]
            assert len(actions)>=2
    return x
def lmargin(lp,actions,good,bad):
    return lp[actions.index(good)]-lp[actions.index(bad)]
def objective_loss(policy,group,objective,beta,refs):
    losses=[];deltas=[];records=group["paired_option_records"]
    for j,row in enumerate(records):
        actions=tuple(row["option_order"])
        lp=policy.log_distribution(model_input(row),actions)
        chosen=row["chosen_action"];rejected=row["rejected_action"]
        if objective=="action_ce":
            losses.append(-lp[actions.index(chosen)])
        else:
            d=lmargin(lp,actions,chosen,rejected)
            deltas.append(d-refs[j])
    if objective=="action_ce":
        return torch.stack(losses).mean(),None
    # ONE comparison weighted once, with V4's existing two option orders.
    delta=torch.stack(deltas).mean()
    return -F.logsigmoid(beta*delta),delta
def compute_refs(policy,group):
    ref=[]
    with torch.inference_mode():
        for row in group["paired_option_records"]:
            actions=tuple(row["option_order"])
            lp=policy.reference_log_distribution(model_input(row),actions)
            ref.append(lmargin(lp,actions,row["chosen_action"],row["rejected_action"]).detach())
    return tuple(ref)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--groups",type=Path,required=True)
    ap.add_argument("--base",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--objective",choices=("action_ce","dpo"),required=True)
    ap.add_argument("--release-only",action="store_true")
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--steps",type=int,default=96)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--beta",type=float,default=2.0)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.steps<1:raise ValueError("Invalid steps")
    groups=read_groups(args.groups,args.release_only)
    # Record sizes by ORIGINAL INDEPENDENT COMPARISON, never report 2x swaps
    # as two independent samples.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    policy=HFActionPolicy(
        model_path=args.base,trainable=True,quantization="4bit",dtype="bf16",
        max_prompt_tokens=4096,lora_rank=8,lora_alpha=16,
        gradient_checkpointing=True,allow_download=False,temperature=1.,
        explore_epsilon=0.)
    print("V4_MODEL_LOADED",policy.policy_id,"objective",args.objective,
        "independent_groups",len(groups),"original_rows",sum(x["n_original_records"] for x in groups),
        "roles",dict(Counter(g["role"] for g in groups)),flush=True)
    t=time.monotonic()
    references={}
    if args.objective=="dpo":
        for group in groups:
            references[group["group_id"]]=compute_refs(policy,group)
        print("V4_FROZEN_BASE_REFERENCE_READY",len(references),flush=True)
    optim=torch.optim.AdamW(policy.trainable_parameters(),lr=args.lr,weight_decay=0.)
    curve=[]
    rng=random.Random(args.seed)
    order=[]
    seen_counts=Counter()
    for step in range(args.steps):
        if not order:
            order=list(range(len(groups)))
            rng.shuffle(order)
        g=groups[order.pop()]
        policy.model.train()
        optim.zero_grad(set_to_none=True)
        ref=references.get(g["group_id"])
        loss,delta=objective_loss(policy,g,args.objective,args.beta,ref)
        assert torch.isfinite(loss)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(),1.,
                                            error_if_nonfinite=True)
        assert torch.isfinite(grad)
        optim.step()
        seen_counts[g["group_id"]]+=1
        curve.append({
            "update":step+1,"comparison_id":g["group_id"],
            "root":g["root"],"role":g["role"],"stage":g["stage"],
            "category":g["category"],"original_option_order_rows":2,
            "loss":float(loss.detach()),
            "reference_relative_margin":float(delta.detach()) if delta is not None else None,
            "preclip_gradient_norm":float(grad),
        })
        if step==0 or (step+1)%12==0:
            print("V4_4B_TRAIN",args.objective,step+1,
                  "loss",round(float(loss.detach()),6),
                  "rel_margin",round(float(delta.detach()),5) if delta is not None else "CE",
                  "grad",round(float(grad),5),flush=True)
    args.output.mkdir(parents=True)
    policy.save_adapter(args.output/"adapter")
    result={
        "status":"TRAINED_REAL_QWEN3_4B_V4_NATIVE_PREFS",
        "action_scoring_format":FORMAT,
        "constant_output_instruction":ACTION_INSTRUCTION,
        "one_fresh_trainable_LoRA":True,
        "role_specific_LoRA_switches":False,
        "role_and_stage_native_from_data":True,
        "original_normal_swapped_rows_not_regenerated":True,
        "objective":args.objective,"seed":args.seed,
        "steps":args.steps,"optimizer":"AdamW",
        "lr":args.lr,"beta":args.beta if args.objective=="dpo" else None,
        "reference_is_same_frozen_pretrained_base":args.objective=="dpo",
        "reference_is_not_native_Qstar_or_actual_model_Qpi":True,
        "no_native_oracle_reference_in_LM_prompt":True,
        "no_DG36_processed":True,
        "group_source_file_sha256":sha256(args.groups.read_bytes()).hexdigest(),
        "base_model":args.base,"quantization":"nf4_bf16",
        "lora_rank":8,"lora_alpha":16,
        "original_record_count":sum(g["n_original_records"] for g in groups),
        "independent_comparison_count":len(groups),
        "independent_training_roots":sorted({g["root"] for g in groups}),
        "by_role":dict(Counter(g["role"] for g in groups)),
        "by_stage":dict(Counter(g["stage"] for g in groups)),
        "by_category":dict(Counter(g["category"] for g in groups)),
        "updates_per_comparison":dict(seen_counts),
        "reference_margins_by_group":{k:[float(v) for v in values]
                                        for k,values in references.items()},
        "wall_seconds":round(time.monotonic()-t,2),
        "training_curve":curve,
    }
    (args.output/"TRAINING.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("V4_TRAIN_FINISHED",args.objective,"steps",args.steps,
          "time_sec",round(time.monotonic()-t,2),flush=True)
if __name__=="__main__":main()
