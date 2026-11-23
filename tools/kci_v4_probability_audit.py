"""V4 original vs swapped RAW/VERIFY model-probability evidence audit."""
from pathlib import Path
from collections import defaultdict
import json,math
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/20261123_v4_release_control_v1"
M=("frozen_base","existing_frozen_R","v4_action_ce","v4_dpo")
R="SEND_RAW_R";V="VERIFY_PRIVATE_R"
def mean(xs):return round(sum(xs)/len(xs),8) if xs else None
def main():
    records=[json.loads(x) for x in (BASE/"eval_release_4arms_s7/MODEL_DECISIONS_ALL.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
    assert len(records)==19
    full={}
    for split in ("train","dev"):
        x=[r for r in records if r["split"]==split]
        full[split]={}
        for mode in M:
            row={}
            for cat,items in (
                 ("risky",[r for r in x if r["condition"]=="risky"]),
                 ("legitimate",[r for r in x if r["condition"]!="risky"])):
                assert items
                row[cat]={
                   "n_independent_states":len(items),
                   "Praw_original_mean":mean([r["methods"][mode]["P_normal"][R] for r in items]),
                   "Praw_swapped_mean":mean([r["methods"][mode]["P_swapped"][R] for r in items]),
                   "Pverify_original_mean":mean([r["methods"][mode]["P_normal"][V] for r in items]),
                   "Pverify_swapped_mean":mean([r["methods"][mode]["P_swapped"][V] for r in items]),
                   "semantic_action_flips":sum(r["methods"][mode]["position_semantic_flip"] for r in items),
                }
            row["history_conditioned_raw_probability_separation_original"] = (
               row["legitimate"]["Praw_original_mean"]-row["risky"]["Praw_original_mean"])
            row["history_conditioned_raw_probability_separation_swapped"] = (
               row["legitimate"]["Praw_swapped_mean"]-row["risky"]["Praw_swapped_mean"])
            full[split][mode]=row
    out=BASE/"eval_release_4arms_s7/PROBABILITY_AUDIT.json"
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(full,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(full,ensure_ascii=False,indent=2)[:11000])
if __name__=="__main__":main()
