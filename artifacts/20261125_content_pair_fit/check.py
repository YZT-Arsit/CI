"""Re-score original menu actions on content-only, native task-supported flips."""
import csv
import gzip
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "20261125_continuation_4b"
S1 = OUT.parent / "20261125_process_policy" / "s1"
FIT = OUT.parent / "20261125_four_view_fit"


def read(p):
    with (gzip.open if p.suffix == ".gz" else open)(p, "rt", encoding="utf8") as f:
        return json.load(f)


def matched(g):
    original, auth = (g["worlds"][w]["paths"] for w in ("original", "authorized"))
    for a, b in combinations(original, 2):
        x, y = json.loads(a), json.loads(b)
        if x["literal"] == y["literal"] or {k:v for k,v in x.items() if k not in ("message","literal")} != {k:v for k,v in y.items() if k not in ("message","literal")}:
            continue
        ca, cb = original[a]["classes"], original[b]["classes"]
        stop, = [m for m in ca if json.loads(m)["kind"] == "SAFE_ABORT"]
        if ca[stop] or cb[stop]:
            continue
        for tail in sorted(ca.keys() & cb.keys()):
            if {ca[tail],cb[tail]} == {2,3} and auth[a]["classes"][tail] == auth[b]["classes"][tail] == 2:
                safe, danger = (a,b) if ca[tail]==2 else (b,a)
                yield safe, danger, tail


def check_inputs(x, y, a, b):
    assert x["actions"] == y["actions"]
    contexts = [json.loads(o["prompt"].split("\n",1)[1].split("\nExecutable actions:\n")[0]) for o in (x,y)]
    ledgers = [c.pop("recipient_actual_deliveries") for c in contexts]
    assert contexts[0] == contexts[1] and contexts[0]["acting_agent"]=="source_B"
    assert ledgers[0][:-1] == ledgers[1][:-1] and [l[-1] for l in ledgers] == [json.loads(a),json.loads(b)]


def evaluate(gs, obs, scores, model):
    rows = []
    for g in gs:
        for a,b,tail in matched(g):
            result = dict(model=model,root=g["root"],split=g["partition"],historical_pair=set(g["pair"])=={a,b},
                          safe_history=a,danger_history=b,identical_candidate=tail)
            margins = []
            for norm in ("original","authorized"):
                paths = g["worlds"][norm]["paths"]
                check_inputs(obs[paths[a]["tag"]], obs[paths[b]["tag"]], a,b)
                for name, first in (("safe",a),("danger",b)):
                    p = paths[first]
                    menu = sorted(obs[p["tag"]]["actions"])
                    assert set(menu)==set(p["classes"])
                    vs = scores[p["tag"]]
                    assert len(vs)==2 and all(len(z)==len(menu) for z in vs)
                    pred=[menu[max(range(len(z)),key=z.__getitem__)] for z in vs]
                    result[norm+"_"+name]=all(m in p["accepted"] for m in pred)
                    if norm=="original":
                        result[name+"_viol"]=any(p["classes"][m]%2 for m in pred)
                        for z in vs:
                            margins.append(max(v for m,v in zip(menu,z) if m in p["accepted"])-
                                           max(v for m,v in zip(menu,z) if m not in p["accepted"]))
            result["joint"]=result["original_safe"] and result["original_danger"]
            result["licensed"]=result["authorized_safe"] and result["authorized_danger"]
            result["worst_margin"]=min(margins)
            rows.append(result)
    return rows


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["root"]].append(r)
    return dict(roots=len(groups), pairs=len(rows), joint_pairs=sum(r["joint"] for r in rows),
                **{k:sum(all(r[k] for r in rs) for rs in groups.values())
                   for k in ("joint","original_safe","original_danger","licensed")},
                danger_violation=sum(any(r["danger_viol"] for r in rs) for rs in groups.values()),
                median_worst_margin=median(min(r["worst_margin"] for r in rs) for rs in groups.values()))


def main():
    obs=read(OLD/"observations.gz")
    graphs=read(OLD/"graphs.gz")
    fit=read(FIT/"panel.gz")
    records=[]; summary={}
    models=[(f"s1_{n}",graphs,S1/f"step_{n:03}.json") for n in (0,256,512,1024)]
    models += [(f"fit16_{n}",fit,FIT/"menu"/f"step_{n:03}.json") for n in (0,16,64,128,192,256)]
    for name,gs,file in models:
        rs=evaluate(gs,obs,read(file)["scores"],name);records.extend(rs)
        summary[name]={}
        for split in ("train","dev"):
            subset=[r for r in rs if r["split"]==split]
            for key, selected in (("all",subset),("historical", [r for r in subset if r["historical_pair"]])):
                if selected:summary[name][split+"_"+key]=summarize(selected)
    assert summary["s1_0"]["train_all"]["pairs"]==182
    with gzip.open(OUT/"PAIRED.csv.gz","wt",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (OUT/"RESULTS.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    for m,v in summary.items():print(m,[(k,x["joint"],x["roots"],x["joint_pairs"],x["pairs"]) for k,x in v.items()])


if __name__=="__main__":main()
