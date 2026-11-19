"""Paired curriculum selector with no model calls or labels invented.
Select equal numbers of real flip and hold pairs, matching role/task where possible.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def make(out):
    qs={r["id"]:r for r in rows(out/"queries.jsonl")}
    pairs=rows(out/"pairs.jsonl")
    flips=[p for p in pairs if p["split"]=="train" and p["relation_flip"]]
    holds=[p for p in pairs if p["split"]=="train" and not p["relation_flip"]]
    if not flips:raise ValueError("No train flip supervision")
    used=set();selected=[];kind_counts=Counter();hold_label_counts=Counter()
    for p in flips:
        choices=[h for h in holds if h["role"]==p["role"] and h["base_group_id"] not in used]
        if not choices:raise ValueError("insufficient hold controls")
        def rank(h):
            f,g=p["factors"],h["factors"]
            left,right=qs[h["left_id"]]["label"],qs[h["right_id"]]["label"]
            if left!=right:raise AssertionError("HOLD pair has inconsistent labels")
            return (
                hold_label_counts[left],
                h["base_group_id"]!=p["base_group_id"],
                f["kind"]!=g["kind"],
                f["depth"]!=g["depth"],
                f["allow"]!=g["allow"],
                f["required"]!=g["required"],
                f["delivered"]!=g["delivered"],
                h["base_group_id"],h["state_index"],h["action"])
        h=min(choices,key=rank)
        used.add(h["base_group_id"])
        hold_label_counts[qs[h["left_id"]]["label"]]+=1
        selected.extend((p,h))
        kind_counts[(p["role"],"flip")]+=1
        kind_counts[(h["role"],"hold")]+=1
    seen=set();records=[];counts=Counter()
    for p in selected:
        for side in ("left_id","right_id"):
            r=qs[p[side]]
            # Some hold roots could be selected more than once through different actions.
            # The selection is sample-level and no row is allowed to be counted twice.
            if r["id"] in seen:raise AssertionError("duplicate sample across selected pairs")
            seen.add(r["id"])
            records.append({**r,"curriculum_class":"flip" if p["relation_flip"] else "hold",
                            "curriculum_pair_id":p["base_group_id"]+":"+str(p["state_index"])+":"+p["action"],
                            "training_ready":False})
            counts[r["label"]]+=1
    if len(records)!=4*len(flips):raise AssertionError("paired curriculum shape")
    p=out/"balanced_train_candidate.jsonl"
    if p.exists():raise FileExistsError(p)
    p.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in records),encoding="utf-8")
    report={"selected_pairs":len(selected),"flip_pairs":len(flips),"hold_pairs":len(flips),
            "candidate_rows":len(records),"positive":counts[1],"negative":counts[0],
            "roles":{str(k):v for k,v in sorted(kind_counts.items())},
            "label_provenance":"native exact action continuation; not GPT generated",
            "data_exposure":"TRAIN split only; DEV not read for selection",
            "trained_or_verified_model_gain":False,
            "independent_business_diversity":False,"training_ready":False}
    (out/"curriculum_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":make(Path(sys.argv[1]))
