"""Strict no-lexicographic-tie audit for frozen V4 semantic 4B outputs.

No weights, source labels, model actions or original evaluation files are
modified. Exact equality among top native-action logits is UNRESOLVED.
This audits the failure to distinguish strict preference from argmax order.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter
import json,sys
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261123_v4_paired_repair_v1"
R="SEND_RAW_R";V="VERIFY_PRIVATE_R"
def jl(path):return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
def exact_strict(scores,chosen):
    return all(scores[chosen]>s for action,s in scores.items() if action!=chosen)
def stable_action(scores):
    winners=[a for a,v in scores.items() if v==max(scores.values())]
    return winners[0] if len(winners)==1 else "UNRESOLVED_TIE"
def summarize(rows):
    byroot=defaultdict(dict)
    for x in rows:
        byroot[x["root"]][x["condition"]]=x
    return {
       "n_states":len(rows),"n_roots":len(byroot),
       "strictly_correct_states":sum(x["strictly_correct"] for x in rows),
       "exact_top_tie_states":sum(x["predicted_strict"]=="UNRESOLVED_TIE" for x in rows),
       "strict_flip_pairs":sum(v["risky"]["strictly_correct"] and v["safe"]["strictly_correct"] for v in byroot.values()),
       "strict_authorized_pairs":sum(v["risky"]["strictly_correct"] and v["authorized"]["strictly_correct"] for v in byroot.values()),
       "strict_root_triplets":sum(all(x["strictly_correct"] for x in v.values()) for v in byroot.values() if {"risky","safe","authorized"}<=set(v)),
       "per_root":{k:{n:{"strictly_correct":x["strictly_correct"],
                         "strict_prediction":x["predicted_strict"],
                         "gold":x["gold"],
                         "RAW_minus_VERIFY":x["scores"][R]-x["scores"][V],
                         "Praw":x["probabilities"][R]}
                       for n,x in v.items()} for k,v in byroot.items()},
    }
def main():
    source=jl(D/"train_gate_grouped_ce/DECISIONS_TRAIN_ONLY.jsonl")
    train=[]
    for r in source:
        scores=r["candidate_scores"]
        train.append({"root":r["root"],"condition":r["condition"],"gold":r["gold"],
            "scores":scores,"probabilities":r["probabilities"],
            "predicted_strict":stable_action(scores),
            "strictly_correct":exact_strict(scores,r["gold"])})
    dev_source=jl(D/"dev_eval_grouped_ce/DECISIONS_DG34_DG35.jsonl")
    dev={}
    for view in ("native","entity_alias"):
        rows=[]
        for r in dev_source:
            pred=r["views"][view]
            scores=pred["candidate_scores"]
            gold=r["native_truth"]
            rows.append({"root":r["root"],"condition":r["condition"],"gold":gold,
               "scores":scores,"probabilities":pred["P"],
               "predicted_strict":stable_action(scores),
               "strictly_correct":exact_strict(scores,gold)})
        dev[view]=summarize(rows)
    out={"policy":"exact top-score tie is UNRESOLVED not semantically correct",
       "source_immutable":True,"seed":7,"n_original_training_roots":3,
       "original_train_argmax_9_of_9_has_tie":True,
       "corrected_train_gate_strict":False,
       "DG34_DG35_dev_were_already_evaluated_with_old_tie_aware_gate":True,
       "train":summarize(train),"dev":dev,
       "relabel_or_overwrite_original_metrics":False,
       "do_not_claim_dev_as_independent_blind_test":True}
    result=D/"STRICT_TIE_AUDIT.json"
    if result.exists():raise FileExistsError(result)
    result.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("TIE_AUDIT",json.dumps({"train":{k:v for k,v in out["train"].items() if k!="per_root"},
       "dev":{name:{k:v for k,v in values.items() if k!="per_root"} for name,values in dev.items()}},
       ensure_ascii=False))
if __name__=="__main__":main()
