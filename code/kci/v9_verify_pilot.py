"""Rebuild native rewards from delivered sources and replay sealed model decisions."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import mean
import argparse
import json
import math

from .paper_worlds import key
from .v9_execution import Execution, read_gz, write_gz
from .v9_interface import Deliveries, sha
from .v9_schedule import schedule


def verify_run(run, subset, source, prompts, traces):
    rows = run["root_results"]
    assert {r["id"] for r in rows} == set(subset)
    for row in rows:
        g = source[row["id"]]
        def selected(tag):
            scores = run["scores"][tag]
            actions = prompts[tag]["actions"]
            assert len(actions) == len(scores)
            return actions[max(range(len(scores)), key=scores.__getitem__)]
        assert selected(g["first_input"]) == row["first"]
        first = g["first_input"]
        pair = [g["chosen"], g["rejected"]]
        assert pair == row["weight_pair"]
        margins = [run["scores"][first][prompts[first]["actions"].index(a)] for a in pair]
        assert margins[0] - margins[1] == row["upstream_margin"]
        for a, bs in g["paths"].items():
            native_ref = sum(Fraction(b["probability"])*b["reference_stsr"] for b in bs)
            assert abs(float(native_ref) - row["qref"][a]) < 1e-12
            totals = Counter()
            for b, logged in zip(bs, row["events"][a], strict=True):
                tag = b["policy_input"]
                act = selected(tag)
                actual = b["outcomes"][act]
                mass = Fraction(b["probability"])
                assert (tag, act, b["condition"], str(mass)) == (logged["input"], logged["action"], logged["condition"], logged["mass"])
                for k in ("success", "direct", "inferred", "stsr", "abort"):
                    assert actual[k] == logged[k]
                    totals[k] += mass*actual[k]
                if a == row["first"]:
                    deliveries=sum(e.get("status")=="DELIVERED" for e in actual["events"])
                    if row["domain"]=="proof":
                        deliveries-=sum(e["action"]["actor"]=="historical_source" and e["status"]=="DELIVERED"
                                        for e in actual["events"])
                    traces.append(dict(id=row["id"], weights=run["weights"], split=row["split"], event=b["condition"],
                                       mass=str(mass), first=a, first_input=first, downstream=act, downstream_input=tag,
                                       new_signed_deliveries=deliveries, **actual))
            assert all(abs(float(totals[k])-row["qpi"][a][k]) < 1e-12 for k in totals)
        assert row["actual"] == row["qpi"][row["first"]]
        assert row["qpi_gap"] == row["qpi"][pair[0]]["stsr"] - row["qpi"][pair[1]]["stsr"]


def metrics(rows, initial, planning):
    base = {r["id"]: r for r in initial}
    n = len(rows)
    def avg(f):
        return mean(f(r) for r in rows)
    def any_risk(r):
        return float(sum(Fraction(v["mass"]) * (v["direct"] or v["inferred"]) for v in r["events"][r["first"]]))
    def pair(r, i):
        return r["weight_pair"][i]
    regrets = []
    for r in rows:
        values = planning[r["id"]]
        if r["first"] in values:
            regrets.append(max(values.values()) - values[r["first"]])
    return dict(roots=n, immediately_legal_roots=len(regrets),
                exact_safe_success_regret=mean(regrets) if regrets else None,
                optimal_safe_success=avg(lambda r:max(planning[r["id"]].values())),
                attainable_after_actual_first=avg(lambda r:planning[r["id"]].get(r["first"],0.)),
                actual_first_outside_labeled_pair=sum(r["first"] not in r["weight_pair"] for r in rows),
                **{k:avg(lambda r:r["actual"][k]) for k in ("success","stsr","direct","inferred","abort")},
                any_violation=avg(any_risk), upstream_correct=sum(r["rank"] for r in rows),
                upstream_ties=sum(r["upstream_margin"] == 0 for r in rows), qpi_correct=sum(r["value_rank"] for r in rows),
                qpi_ties=sum(r["value_tie"] for r in rows), previously_masked=sum(base[r["id"]]["qpi_gap"]<=0 for r in rows),
                restored=sum(base[r["id"]]["qpi_gap"]<=0<r["qpi_gap"] for r in rows),
                lost=sum(r["qpi_gap"]<=0<base[r["id"]]["qpi_gap"] for r in rows),
                qpi_pos=avg(lambda r:r["qpi"][pair(r,0)]["stsr"]),qpi_neg=avg(lambda r:r["qpi"][pair(r,1)]["stsr"]),
                qref_pos=avg(lambda r:r["qref"][pair(r,0)]),qref_neg=avg(lambda r:r["qref"][pair(r,1)]),
                legal_disclosure_retention=avg(lambda r:r["legal_disclosure_retention"]))


def summarize(out, delivery):
    load=lambda p:json.loads(p.read_text(encoding="utf8"))
    frozen=load(out/"freeze.json")
    assert all(sha(out/k)==v for k,v in frozen["hashes"].items())
    d=Deliveries(delivery/"deliverables/KCI_VALIDATED_V9_INTERNAL",delivery/"d_future_core_20261124/D_future_core_multisource_v2")
    by={r["id"]:r for r in frozen["groups"]}
    records=[r for name in ("proofwriter_local_attestation","negotiation")
             for r in d.core_reference(name,purpose="reference_distillation") if r["row_id"] in by]
    assert len(records)==len(by)
    ex=Execution(d,records)
    source={r["row_id"]:ex.graph(r) for r in records}
    prompts=read_gz(out/"prompts.gz")
    assert ex.prompts==prompts
    gs=read_gz(out/"graphs.gz")
    planning={};mechanism={}
    for g in gs:
        orig=source[g["id"]]
        planning[g["id"]]={}
        for a,branches in g["paths"].items():
            if g["valid_first"][a]:
                # One remaining decision after each observed chance event. Safe abstention is always available.
                planning[g["id"]][a]=float(sum(Fraction(b["probability"])*max(o["stsr"] for o in b["outcomes"].values())
                                                 for b in orig["paths"][a]))
            for b,nb in zip(branches,orig["paths"][a],strict=True):
                assert all(b[k]==nb[k] for k in ("reference_stsr","condition","policy_input","probability"))
                assert b["outcomes"].keys()==nb["outcomes"].keys()
                for act, o in b["outcomes"].items():
                    assert all(nb["outcomes"][act][k]==v for k,v in o.items())
        pair=[g["chosen"],g["rejected"]]
        if g["domain"]=="proof":
            env=ex.proofs[g["id"]]
            authorized=ex.pw.Env(env.src,env.spec,env.mode,licensed=True)
            vals=[env.run(json.loads(a)) for a in pair]
            licensed=[authorized.run(json.loads(a))["safe_success"] for a in pair]
            assert all(v["first_audit"]["verdict"]=="VALID" for v in vals)
            assert vals[0]["first_K"]!=vals[1]["first_K"]
            by_action=[{key(o["action"]):o for o in v["downstream"]} for v in vals]
            flips=sum(o["audit"]["verdict"]!=by_action[1][k]["audit"]["verdict"] and
                      o["all_task_claims_verifiable"] and by_action[1][k]["all_task_claims_verifiable"]
                      for k,o in by_action[0].items())
            assert flips and [v["safe_success"] for v in vals]==[1,0]
        else:
            witness=ex.neg[g["id"]]
            licensed=[float(sum(Fraction(1,7)*b["norm_ablated"]["task_success"] for b in
                      next(x["continuations"] for x in witness["branches"] if x["first_delivery"]["content"]==json.loads(a)))) for a in pair]
            flips=None
        assert licensed[0]==licensed[1]
        mechanism[g["id"]]=dict(norm_ablated=licensed,verified_matched_disclosure_flips=flips,
                                 exact_safe_values=planning[g["id"]],label_pair=pair,
                                 reference=g["reference_stsr"])
    results={};exposures={};gates={};traces=[]
    init={s:load(out/(s+"_initial.json")) for s in ("proof","neg")}
    for domain,r in init.items():
        assert r["weights"]==frozen["initial"]
        groups=[i for i,v in by.items() if v["domain"]==domain and v["split"]!="interface"]
        verify_run(r,groups,source,prompts,traces)
        dev=[v for v in r["root_results"] if v["split"]=="dev"]
        results[domain+"_initial"]=metrics(dev,dev,planning)
    for sched in sorted(out.glob("schedule_*.json")):
        s=load(sched); seed=s["seed"]
        assert s==schedule(gs,init,seed)
        for arm,rows in s["arms"].items():
            tag=f"{arm}_{seed}";p=out/tag
            if not (p/"result.json").exists():continue
            r=load(p/"result.json")
            if not rows:
                assert r["updates"]==0
                continue
            assert r["updates"]==len(rows) and r["start"]==frozen["initial"] and r["schedule"]==sha(sched)
            logs=[json.loads(x) for x in (p/"updates.jsonl").read_text().splitlines()]
            assert len(logs)==len(rows)
            assert all(all(l[k]==v for k,v in row.items()) for l,row in zip(logs,rows))
            assert all(l["schedule"]==sha(sched) and l["update"]==i+1 for i,l in enumerate(logs))
            assert all(math.isfinite(l["loss"]) and math.isfinite(l["grad_norm"]) for l in logs)
            if logs[0]["neg"]:
                assert abs(logs[0]["ref_margin"]-logs[0]["margin"])<1e-5
            domain=arm.split("_")[0]
            dev=[v for v in init[domain]["root_results"] if v["split"]=="dev"]
            verify_run(r,[v["id"] for v in dev],source,prompts,traces)
            results[tag]=metrics(r["root_results"],dev,planning)
            exposures[tag]=dict(updates=len(logs),common=len(rows),intervention=len(rows),
                                roots=len({l["root"] for l in logs}),groups=len({l["group"] for l in logs}),
                                max_exposures_per_root=max(Counter(l["root"] for l in logs).values()),
                                action_terms=sum(len(l["common"])+len(l["pos"])+len(l["neg"]) for l in logs),
                                unique_interventions=len({key([l["pos"],l["neg"]]) for l in logs}),
                                checkpoint=r["weights"],schedule=r["schedule"])
        targets=["proof_ci","proof_sft","proof_terminal","proof_cce"]
        if all(f"{a}_{seed}" in results for a in targets):
            *controls,cce=[results[f"{a}_{seed}"] for a in targets]
            best=max(c["upstream_correct"] for c in controls)
            limits=[c["any_violation"] for c in controls if c["upstream_correct"]==best]
            gates[f"proof_{seed}"]=(cce["upstream_correct"]>=best+2 and cce["stsr"]>=max(c["stsr"] for c in controls)
                                    and cce["any_violation"]<=min(limits))
        if all(f"{a}_{seed}" in results for a in ("neg_dpr","neg_ordinary")):
            a,b=results[f"neg_dpr_{seed}"],results[f"neg_ordinary_{seed}"]
            gates[f"neg_{seed}"]=a["restored"]>=b["restored"]+2 and a["stsr"]>b["stsr"] and a["any_violation"]<=b["any_violation"]
    fit_replay={}
    extra=out/"fit_error_replay.json"
    if extra.exists():
        sealed=load(out/"schedule_7.json")
        eligible={x["id"] for x in sealed["arms"]["neg_dpr"]}
        for arm,run in load(extra).items():
            assert exposures[arm+"_7"]["checkpoint"]==run["weights"]
            verify_run(run,eligible,source,prompts,[])
            observed=[]
            for row in run["root_results"]:
                for fail in sealed["downstream_failures"][row["id"]]:
                    if not fail["alternatives"]:continue
                    event=next(v for vs in row["events"].values() for v in vs if v["input"]==fail["input"])
                    observed.append(dict(id=row["id"],input=event["input"],original_error=fail["action"],
                                         after=event["action"],stsr=event["stsr"],inferred=event["inferred"]))
            fit_replay[arm]=observed
    for name,m in results.items():
        weights=exposures[name]["checkpoint"] if name in exposures else frozen["initial"]
        sample=[x for x in traces if x["weights"]==weights and x["split"]=="dev" and
                by[x["id"]]["domain"]==name.split("_")[0] and x["stsr"]]
        mass=sum(Fraction(x["mass"]) for x in sample)
        m["signed_deliveries_per_safe_success"]=(float(sum(Fraction(x["mass"])*x["new_signed_deliveries"]
                                                          for x in sample)/mass) if mass else None)
    write_gz(out/"replayed_deliveries.json.gz",traces)
    write_gz(out/"mechanism_witnesses.json.gz",mechanism)
    report=dict(metrics=results,exposure=exposures,gates=gates,fit_error_replay=fit_replay,native_roots_checked=len(gs),
                policy_observations_checked=len(prompts),replayed_autonomous_events=len(traces))
    (out/"RESULTS.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(gates)


if __name__=="__main__":
    arg=argparse.ArgumentParser(__doc__)
    arg.add_argument("--out",type=Path,required=True)
    arg.add_argument("--delivery",type=Path,required=True)
    a=arg.parse_args();summarize(a.out,a.delivery)

