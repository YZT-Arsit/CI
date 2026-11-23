"""Train-only V4 root-grouped semantic scorer validation.

Never opens DG34/35/DG36 or native dev labels. Freezes the trained 4B
adapter and evaluates original 9 TRAIN states. Decides next diagnostic
ONLY from TRAIN performance. Does not select any model by held-out dev.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter
from hashlib import sha256
import json,sys,time,argparse
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_v4_semantic_action import scoring,assert_views_position_invariant
from kci_v4_pairwise_repair_train import group_from_verified_states
RAW,VERIFY="SEND_RAW_R","VERIFY_PRIVATE_R"
def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=Path,required=True)
    ap.add_argument("--base",required=True)
    ap.add_argument("--adapter",required=True)
    ap.add_argument("--training",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    assert not args.output.exists()
    spec=json.loads(args.training.read_text(encoding="utf8"))
    assert spec["training_data_sha256"]==sha(args.train)
    assert spec["no_dev_or_final_test_used_in_training"]
    groups=group_from_verified_states(args.train)
    hf=HFActionPolicy(model_path=args.base,adapter=args.adapter,trainable=False,
        quantization="4bit",dtype="bf16",max_prompt_tokens=4096,allow_download=False)
    hf.model.eval()
    result=[];start=time.monotonic()
    with torch.inference_mode():
        for root,rows in groups.items():
            for cond in ("risky","safe","authorized"):
                row=rows[cond]
                assert assert_views_position_invariant(
                    row["prompt"],row["original_action_order"],
                    row["swapped_prompt"],row["swapped_action_order"])
                acts,scores=scoring(hf,row["prompt"],row["original_action_order"])
                probs=torch.softmax(scores.float(),dim=0).cpu().tolist()
                p=dict(zip(acts,[float(x) for x in probs]))
                s=dict(zip(acts,[float(x) for x in scores.float().cpu().tolist()]))
                pred=max(p,key=p.get)
                rec={"root":root,"condition":cond,"role":row["role"],
                  "native_state_hash":row["native_state_hash"],
                  "source_comparison_id":row["comparison_id"],
                  "gold":row["chosen_action"],"predicted":pred,
                  "correct":pred==row["chosen_action"],
                  "probabilities":p,"candidate_scores":s,
                  "raw_minus_verify":s[RAW]-s[VERIFY],
                  "one_position_invariant_semantic_input":True}
                result.append(rec)
                print("V4_REPAIR_TRAIN_PRED",spec["method"],root,cond,
                      pred,"Praw",round(p[RAW],5),"Pverify",round(p[VERIFY],5),flush=True)
    by_root=defaultdict(dict)
    for x in result:by_root[x["root"]][x["condition"]]=x
    pairs={}
    for root,cond in by_root.items():
        pairs[root]={
          "Flip":cond["risky"]["predicted"]==VERIFY and cond["safe"]["predicted"]==RAW,
          "authorization":cond["risky"]["predicted"]==VERIFY and cond["authorized"]["predicted"]==RAW,
          "risk_safe_authorized_all_correct":all(x["correct"] for x in cond.values())}
    n_flip=sum(x["Flip"] for x in pairs.values())
    n_all=sum(x["risk_safe_authorized_all_correct"] for x in pairs.values())
    summary={"status":"TRAIN_ONLY_REPAIR_GATE",
       "model_method":spec["method"],
       "train_data_sha256":sha(args.train),
       "model_adapter":args.adapter,
       "training_meta_sha256":sha(args.training),
       "root_count":3,"independent_states":9,
       "gold_correct":sum(x["correct"] for x in result),
       "true_flip_both_sides_correct_roots":n_flip,
       "root_triplets_all_correct":n_all,
       "true_authorization_pair_correct_roots":sum(x["authorization"] for x in pairs.values()),
       "per_root":pairs,
       "fit_gate_passed":n_all==3,
       "DG34_DG35_DG36_NOT_READ":True,
       "no_oracle_Qpi_claim":True,
       "inference_candidate_forward_count":len(result)*3,
       "inference_wall_seconds":round(time.monotonic()-start,2)}
    args.output.mkdir(parents=True)
    (args.output/"DECISIONS_TRAIN_ONLY.jsonl").write_text("".join(
         json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in result),encoding="utf8")
    (args.output/"SUMMARY_TRAIN_ONLY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_REPAIR_TRAIN_GATE",json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
