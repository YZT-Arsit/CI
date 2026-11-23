"""Inspect true state-dependent probability margins behind strict action choices."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse,json,math
RAW,VERIFY="SEND_RAW_R","VERIFY_PRIVATE_R"
def l(x):return math.log(max(float(x),1e-30))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--decisions",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    assert not args.output.exists()
    rows=[json.loads(s) for s in args.decisions.read_text(encoding="utf8").splitlines() if s.strip()]
    paired=defaultdict(dict)
    for row in rows:paired[row["world"]][row["branch"]]=row
    assert len(paired)==28
    out=[]
    methods=list(rows[0]["predictions"])
    for world,by in paired.items():
        assert set(by)=={"risky","safe"}
        r,s=by["risky"],by["safe"]
        metrics={}
        for model in methods:
            pr=r["predictions"][model]["prob_balanced"]
            ps=s["predictions"][model]["prob_balanced"]
            mr=l(pr[RAW])-l(pr[VERIFY])
            ms=l(ps[RAW])-l(ps[VERIFY])
            metrics[model]={
                "raw_prob_risky":pr[RAW],
                "raw_prob_safe":ps[RAW],
                "raw_minus_verify_logodds_risky":mr,
                "raw_minus_verify_logodds_safe":ms,
                "history_conditioned_delta_safe_minus_risky":ms-mr,
            }
        out.append({"world":world,"split":r["split"],"root":r["root"],
                    "is_true_two_way":r["true_label"]==VERIFY and s["true_label"]==RAW,
                    "models":metrics})
    group={}
    for split in ("train","dev","exploratory_test"):
        relevant=[r for r in out if r["split"]==split and r["is_true_two_way"]]
        group[split]={"n_worlds":len(relevant),"n_roots":len(set(r["root"] for r in relevant))}
        if relevant:
            group[split]["mean_history_delta"]={m:sum(row["models"][m]["history_conditioned_delta_safe_minus_risky"]
                 for row in relevant)/len(relevant) for m in methods}
            group[split]["mean_raw_prob_safe"]={m:sum(row["models"][m]["raw_prob_safe"]
                 for row in relevant)/len(relevant) for m in methods}
            group[split]["mean_raw_prob_risky"]={m:sum(row["models"][m]["raw_prob_risky"]
                 for row in relevant)/len(relevant) for m in methods}
    summary={"source":str(args.decisions),"groups":group,"per_world":out}
    args.output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(group,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
