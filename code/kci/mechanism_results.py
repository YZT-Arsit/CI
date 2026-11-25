"""Paired root-level readout for the sealed v0.8 draw schedules and native assessments."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
from statistics import mean

from .mechanism_v08 import direction,stream
from .paper_run import dump
from .paper_worlds import key,filehash

SEEDS=(7,1234,2025)
ARMS=("A","B","C","D","E","E_CE")


def per_root(score,flips):
    result=defaultdict(dict)
    for r in score["groups"]:
        root=r["root"]
        result["joint_CI"][root]=int(r["joint"])
        for kind,v in r["correct"].items():result["correct_"+kind][root]=int(v)
        choices={v["kind"] for v in r["choices"].values()}
        for typ in ("SEND_PUBLIC_PROJECTION","SEND_BROKER_PRIVATE","ABORT_SAFE"):
            result["always_"+typ][root]=int(choices=={typ})
    groups=defaultdict(list)
    for r in score["paths"]:groups[r["root"]].append(r)
    for root,rows in groups.items():
        result["optimal_action"][root]=mean(int(r["optimal"]) for r in rows)
        if root in flips:result["flip_joint"][root]=int(all(r["optimal"] for r in rows))
        result["path_joint"][root]=int(all(r["optimal"] for r in rows))
        result["regret_safe"][root]=mean(r["regret"] for r in rows)
        costs=[r["cost_gap"] for r in rows if r["cost_gap"] is not None]
        result["equal_success_share"][root]=len(costs)/len(rows)
        if costs:result["cost_regret_on_equal_success"][root]=mean(costs)
    groups.clear()
    for r in score["system"]["worlds"]:groups[r["root"]].append(r["metrics"])
    for root,rows in groups.items():
        for k in ("stsr","direct","inferred","any_ci","aborted","success","safe_success_cost_mass","unknown","omitted_at_terminal"):
            result[k][root]=mean(r[k] for r in rows)
    return result


def paired(left,right,metric):
    roots=sorted(set.intersection(*(set(v[metric]) for v in [*left,*right])))
    ds=[mean(r[metric][root]-l[metric][root] for l,r in zip(left,right)) for root in roots]
    rng=random.Random(731)
    sims=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(4000))
    return dict(n=len(roots),effect=mean(ds),conditional_root_CI95=[sims[99],sims[3900]],
                positive=sum(d>0 for d in ds),negative=sum(d<0 for d in ds),ties=ds.count(0))


def analyze(out):
    def read(p):return json.loads(p.read_text(encoding="utf8"))
    oracle=defaultdict(list)
    for r in stream(out/"assessment_paths.jsonl.gz"):oracle[r["root"]].append(direction(r,"optimal"))
    flips={root for root,rs in oracle.items() if set(rs)=={-1,1}}
    seal=read(out/"schedule_seal.json")
    exposure={};scores={};means={};per_seed={}
    for arm in ("initial",*ARMS):
        scores[arm]=[]
        for seed in ((0,) if arm=="initial" else SEEDS):
            p=out/f"{arm}_{seed}"
            if seed:
                schedule=out/f"schedule_{seed}_{arm}.jsonl.gz"
                assert filehash(schedule)==seal["sha256"][schedule.name]
                steps=stream(schedule)
                logs=read(p/"optimizer.json")
                assert len(steps)==len(logs["steps"])==seal["steps"]
                assert all(s["row_ids"]==[key(r) for r in batch] for s,batch in zip(logs["steps"],steps))
                assert logs["normalizer"]==4
                data=read(p/"assessment.json")
                assert data["weights"]==logs["updated"] and logs["initial"]==read(out/"initial.json")["weights"]
                draws=[r for batch in steps for r in batch]
                exposure[f"{arm}_{seed}"]=dict(counts=Counter(r["source"] for r in draws),root=Counter(r["root"] for r in draws),
                    intervention_roots=len({batch[2]["root"] for batch in steps}),draws=len(draws))
            else:data=read(out/"initial.json")
            scores[arm].append(per_root(data,flips))
        per_seed[arm]=[{k:mean(v.values()) for k,v in s.items()} for s in scores[arm]]
        for item in per_seed[arm]:item["cost_given_safe_success"]=item["safe_success_cost_mass"]/item["stsr"] if item["stsr"] else None
        means[arm]={k:mean(s[k] for s in per_seed[arm] if s[k] is not None) for k in per_seed[arm][0]}
    for seed in SEEDS:
        steps={arm:stream(out/f"schedule_{seed}_{arm}.jsonl.gz") for arm in ARMS}
        for batches in zip(*(steps[a] for a in ARMS)):
            assert all(b[:2]==batches[0][:2] for b in batches)
            assert len({b[2]["root"] for b in batches})==len({b[3]["root"] for b in batches})==1
            assert all(b[3]==batches[0][3] for b in batches[:4])
            assert batches[3][2]==batches[4][2]==batches[5][2]
            assert batches[4][:3]==batches[5][:3] and dict(batches[4][3],objective="ce")==batches[5][3]
    pairs={}
    for lower,higher in (("A","B"),("A","C"),("A","D"),("B","D"),("C","D"),("D","E"),("E","E_CE")):
        pairs[higher+"-"+lower]={k:paired(scores[lower],scores[higher],k) for k in
                                  ("joint_CI","flip_joint","optimal_action","regret_safe","stsr","any_ci")}
    dump(out/"results.json",dict(strict_flip_roots=sorted(flips),mean=means,per_seed=per_seed,
                                paired=pairs,verified_optimizer_exposures=exposure))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("out",type=Path)
    analyze(p.parse_args().out)

