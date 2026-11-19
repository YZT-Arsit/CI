"""Pre-training shortcut audit of a frozen native KCI composition diagnostic.
No model calls. A baseline intentionally ignores delivered historical facts.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
from hashlib import sha256
import json,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.dynamic_tasks import canonical

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def coarse_action(action):
    return "COMMIT" if action.startswith("COMMIT:") else action
def features(row):
    f=row["factors"]
    return (row["role"],coarse_action(row["action"]),f["kind"],f["depth"],f["allow"],f["required"],f["delivered"])
def score(out):
    q=rows(out/"queries.jsonl");pairs=rows(out/"pairs.jsonl")
    train=[x for x in q if x["split"]=="train"]
    dev=[x for x in q if x["split"]=="dev"]
    lookup={r["id"]:r for r in q}
    keys=defaultdict(Counter)
    for r in train:keys[features(r)][r["label"]]+=1
    majority=Counter(r["label"] for r in train).most_common(1)[0][0]
    def pred(r):
        c=keys.get(features(r))
        return (c.most_common(1)[0][0] if c else majority)
    samples=Counter(); byrole=defaultdict(Counter)
    for r in dev:
        p=pred(r);samples["correct"]+=p==r["label"];samples["total"]+=1
        byrole[r["role"]]["correct"]+=p==r["label"];byrole[r["role"]]["n"]+=1
    pair=Counter();byrole_pair=defaultdict(Counter)
    for p in pairs:
        if p["split"]!="dev":continue
        a,b=lookup[p["left_id"]],lookup[p["right_id"]]
        good=pred(a)==a["label"] and pred(b)==b["label"]
        g="flip" if p["relation_flip"] else "hold"
        pair[g+"_correct"]+=good;pair[g+"_total"]+=1
        byrole_pair[(p["role"],g)]["correct"]+=good
        byrole_pair[(p["role"],g)]["n"]+=1
    if pair["flip_total"]==0:raise AssertionError("DEV has no flipping pairs")
    result={
        "baseline":"history_blind_majority_per_role_action_and_factors",
        "training_examples":len(train),"dev_examples":len(dev),
        "dev_accuracy":samples["correct"]/samples["total"],
        "dev_correct":samples["correct"],
        "dev_total":samples["total"],
        "dev_flip_pair_both_correct":pair["flip_correct"],
        "dev_flip_pairs":pair["flip_total"],
        "dev_hold_pair_both_correct":pair["hold_correct"],
        "dev_hold_pairs":pair["hold_total"],
        "by_role":{k:dict(v) for k,v in sorted(byrole.items())},
        "by_role_pair":{str(k):dict(v) for k,v in sorted(byrole_pair.items())},
        "interpretation":"High individual accuracy is not evidence of relation learning; this baseline deliberately cannot read historical edges.",
        "not_independent_business_generalization":True,
        "no_training_performed":True
    }
    path=out/"shortcut_report.json"
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result
if __name__=="__main__":
    print(json.dumps(score(Path(sys.argv[1])),ensure_ascii=False))
