"""Train-only STRICT V4 4B gate: exact score ties are UNRESOLVED.

Supersedes the older argmax gate which counted an exact RAW/VERIFY tie as
correct by sorted ID ordering. Does not overwrite the old report.
Does NOT load development or final test roots.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,time,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_v4_semantic_action import scoring,assert_views_position_invariant
from kci_v4_pairwise_repair_train import group_from_verified_states,RAW,VERIFY
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def exact_action(scores):
    maximum=max(scores.values())
    winners=[act for act,x in scores.items() if x==maximum]
    return winners[0] if len(winners)==1 else "UNRESOLVED_TIE"
def strict_correct(scores,chosen):
    return all(scores[chosen]>score for act,score in scores.items() if act!=chosen)
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--train",type=Path,required=True)
    a.add_argument("--model",required=True)
    a.add_argument("--adapter",required=True)
    a.add_argument("--training",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    args=a.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    spec=json.loads(args.training.read_text(encoding="utf8"))
    assert spec["training_data_sha256"]==sha(args.train)
    assert spec["no_dev_or_final_test_used_in_training"]
    assert spec["n_independent_training_states"]==9
    groups=group_from_verified_states(args.train)
    model=HFActionPolicy(model_path=args.model,adapter=args.adapter,
        trainable=False,quantization="4bit",dtype="bf16",
        max_prompt_tokens=4096,allow_download=False)
    model.model.eval()
    start=time.monotonic();out=[]
    with torch.inference_mode():
        for root,rows in groups.items():
            for condition in ("risky","safe","authorized"):
                r=rows[condition]
                assert assert_views_position_invariant(r["prompt"],r["original_action_order"],
                    r["swapped_prompt"],r["swapped_action_order"])
                actions,scores=scoring(model,r["prompt"],r["original_action_order"])
                raw_scores={k:float(v) for k,v in zip(actions,scores.detach().float().cpu().tolist())}
                probs=torch.softmax(scores.float(),dim=0).cpu().tolist()
                prediction=exact_action(raw_scores)
                truthful=strict_correct(raw_scores,r["chosen_action"])
                runner_up=max(v for k,v in raw_scores.items() if k!=r["chosen_action"])
                out.append({"root":root,"condition":condition,"role":r["role"],
                    "gold":r["chosen_action"],"prediction":prediction,
                    "strictly_correct":truthful,
                    "exact_gold_margin":raw_scores[r["chosen_action"]]-runner_up,
                    "scores":raw_scores,"probabilities":dict(zip(actions,map(float,probs))),
                    "native_state_hash":r["native_state_hash"],
                    "comparison_id":r["comparison_id"]})
                print("V4_STRICT_TRAIN",spec["method"],root,condition,
                    prediction,"strict",truthful,"margin",raw_scores[r["chosen_action"]]-runner_up,
                    flush=True)
    roots=defaultdict(dict)
    for x in out:roots[x["root"]][x["condition"]]=x
    pair={root:{
       "Flip":d["risky"]["strictly_correct"] and d["safe"]["strictly_correct"],
       "Authorized":d["risky"]["strictly_correct"] and d["authorized"]["strictly_correct"],
       "Triplet":all(x["strictly_correct"] for x in d.values()),
       "min_margin":min(x["exact_gold_margin"] for x in d.values())}
       for root,d in roots.items()}
    report={"method":spec["method"],
        "gate_version":"v2_exact_unique_top_action_required",
        "no_lexicographic_tie_correctness":True,
        "comparison_input_sha256":sha(args.train),
        "n_independent_train_states":len(out),"n_train_roots":3,
        "strict_correct_states":sum(x["strictly_correct"] for x in out),
        "unresolved_exact_ties":sum(x["prediction"]=="UNRESOLVED_TIE" for x in out),
        "strict_Flip_roots":sum(x["Flip"] for x in pair.values()),
        "strict_Authorized_roots":sum(x["Authorized"] for x in pair.values()),
        "strict_Triplet_roots":sum(x["Triplet"] for x in pair.values()),
        "strict_train_gate_passed":all(x["Triplet"] for x in pair.values()),
        "per_root":pair,"DG34_DG35_DG36_never_loaded":True,
        "inference_forwards":3*len(out),
        "wall_seconds":round(time.monotonic()-start,2)}
    args.output.mkdir(parents=True)
    (args.output/"DECISIONS.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in out),encoding="utf8")
    (args.output/"SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_STRICT_TRAIN_GATE",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
