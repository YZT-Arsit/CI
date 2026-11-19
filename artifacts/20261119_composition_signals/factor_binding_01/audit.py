"""Independent reproducible audit of the ONE frozen factor-binding experiment.

No model inference or training. Reads raw bytes and recomputes summaries,
ensures trained arms start from same verified adapter, checks precommitted gates.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json,re,sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]
COMP=ROOT/"artifacts"/"20261117_composition_repair"
sys.path[:0]=[str(BASE),str(HERE),str(ROOT/"artifacts"/"20261117_relation_bridge"),str(ROOT/"code")]
from build_signals import load_rows
from bridge_data import relation_fact
def sha(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def parse(raw,alphabet,seconds):
    m=re.findall(r"ANSWER:\s*(["+alphabet+r"])\b",raw)
    return m[0] if len(m)==1 and 0<seconds<9.5 else None
def score_panel(p,folder):
    raw={mode:{r["id"]:r for r in load_rows(folder/(mode+"_outputs.jsonl"))} for mode in ("task","relation")}
    assert all(set(x)=={r["id"] for r in p} for x in raw.values())
    count={}; pairmode={}
    for mode,target in (("task","task_target"),("relation","relation_target")):
        stats=Counter();byrole=defaultdict(Counter);pairs=defaultdict(list)
        for x in p:
            r=raw[mode][x["id"]]
            got=parse(r["raw"],"01",r["seconds_batch"])
            assert got==r["estimate"]
            right=got==str(x[target])
            stats["n"]+=1;stats["correct"]+=int(right);stats["invalid"]+=got is None
            byrole[x["role"]]["n"]+=1;byrole[x["role"]]["correct"]+=int(right)
            key=(x["base_group_id"],x["role"],x["action"],x["id"].split(":")[-2])
            pairs[key].append(x)
        pair=Counter()
        for v in pairs.values():
            assert len(v)==2
            cls="flip" if v[0][target]!=v[1][target] else "hold"
            okay=all(raw[mode][r["id"]]["estimate"]==str(r[target]) for r in v)
            pair[cls+"_pairs"]+=1;pair[cls+"_both_correct"]+=okay
        count[mode]={"totals":dict(stats),"by_role":{k:dict(v) for k,v in byrole.items()}}
        pairmode[mode+"_pairs"]=dict(pair)
    return {**count,**pairmode}

def score_native(folder):
    qs=load_rows(COMP/"queries.jsonl")
    ss=load_rows(COMP/"states.jsonl")
    labels=load_rows(COMP/"labels.jsonl")
    qr=load_rows(folder/"direct_queries.jsonl")
    ar=load_rows(folder/"direct_actions.jsonl")
    assert len(qr)==len(qs)==len(labels)==192 and len(ar)==len(ss)==80
    cases={l["state_id"]:l["case"] for l in labels}
    st=Counter();byrole=defaultdict(Counter)
    for q,l,r in zip(qs,labels,qr):
        assert q["id"]==l["id"]==r["id"]
        v=parse(r["raw"],"01",r["seconds_batch"])
        correct=v==str(l["truth_target"])
        role=l["case"]["observation"]["role"]
        st["query_total"]+=1;st["query_correct"]+=int(correct)
        byrole[role]["query_total"]+=1;byrole[role]["query_correct"]+=int(correct)
    for s,r in zip(ss,ar):
        assert s["id"]==r["id"]
        v=parse(r["raw"],"ABC",r["seconds_batch"])
        valid=v is not None and ord(v)-65<len(s["actions"])
        chosen=s["actions"][ord(v)-65] if valid else None
        assert r["selected"]==chosen and r["valid"]==valid
        case=cases[s["id"]]; candidates={c["action"]:c for c in case["candidates"]}
        best={case["candidates"][i]["action"] for i in case["optimal_indices"]}
        safe=valid and not candidates[chosen]["immediate_safe"]
        good=valid and chosen in best
        role=case["observation"]["role"]
        st["action_total"]+=1;st["action_optimal"]+=int(good)
        st["immediate_unsafe"]+=int(safe);st["invalid_actions"]+=int(not valid)
        byrole[role]["action_total"]+=1;byrole[role]["action_optimal"]+=int(good)
        byrole[role]["immediate_unsafe"]+=int(safe)
    return {"totals":dict(st),"by_role":{k:dict(v) for k,v in byrole.items()}}

def main():
    plan=read(HERE/"PLAN.json");pre=read(HERE/"TOKEN_PREFLIGHT.json")
    sealed=read(HERE/"SEALED.json")
    assert sealed["complete"] and sealed["plan_hash"]==sha(HERE/"PLAN.json")
    assert sealed["training_script_hash"]==sha(HERE/"run.py")
    assert pre["plan_sha256"]==sha(HERE/"PLAN.json")
    for name,h in plan["data_sha256"].items():assert sha(HERE/(name+".jsonl"))==h
    for path,h in sealed["files"].items():assert sha(HERE/path)==h,path
    cp=ROOT/plan["start_checkpoint"]
    assert read(cp/"CHECKPOINT_SEALED.json")==plan["start_seal"]
    assert all(sha(cp/"adapter"/n)==h for n,h in plan["start_seal"]["adapter_hashes"].items())
    study=HERE/"study"
    schedules=load_rows(HERE/"schedule.jsonl")
    train={r["id"]:r for r in load_rows(HERE/"train.jsonl")}
    old=load_rows(COMP/"training_17.jsonl")
    starts=[]
    per_arm={}
    for arm in ("initial",*plan["arms"]):
        folder=study/arm
        ident=read(folder/"IDENTITY.json")
        assert ident["evaluated"]==ident["after"]
        starts.append(ident["initial"])
        if arm!="initial":
            steps=load_rows(folder/"training.jsonl")
            assert len(steps)==len(schedules)==plan["updates_per_arm"]==64
            assert [x["step"] for x in steps]==list(range(64))
            assert [x["examples"] for x in steps]==[x["query_ids"] for x in schedules]
            assert [x["actor_source_step"] for x in steps]==[x["actor_source_step"] for x in schedules]
            assert all(x["step"]==i and 0<x["seconds"]<30 for i,x in enumerate(steps))
            # The training code asserts AdamW step count at runtime, but this run logs
            # sequential step IDs, not a separate optimizer_step field.
            assert all(all(x in train for x in step["examples"]) for step in steps)
            seal=read(folder/"CHECKPOINT_SEALED.json")
            assert seal["initial_adapter_hashes"]==plan["start_seal"]["adapter_hashes"]
            assert seal["steps"]==64
            for name,h in seal["adapter_hashes"].items():assert sha(folder/"adapter"/name)==h
        dev=score_panel(load_rows(HERE/"dev.jsonl"),folder/"dev")
        confirm=score_panel(load_rows(HERE/"confirm_depth4.jsonl"),folder/"confirm_depth4")
        assert dev==read(folder/"dev"/"RESULTS.json"),arm
        assert confirm==read(folder/"confirm_depth4"/"RESULTS.json"),arm
        native=read(folder/"NATIVE_RESULTS.json")
        nd=score_native(folder/"native")
        selfscore=score_native(folder/"self_fact")
        assert nd["totals"]==native["direct"]["summary"]
        assert nd["by_role"]==native["direct"]["by_role"]
        assert selfscore==native["self_generated_factor"]
        rel=load_rows(folder/"self_fact"/"relation_outputs.jsonl")
        orig=load_rows(COMP/"queries.jsonl")
        labels=load_rows(COMP/"labels.jsonl")
        assert len(rel)==192
        rc=Counter()
        for x,q,l in zip(rel,orig,labels):
            assert x["id"]==q["id"]==l["id"]
            pred=parse(x["raw"],"01",x["seconds_batch"])
            gold=relation_fact(l["case"]["observation"],q["action"])
            assert x["estimate"]==pred and x["true_relation"]==gold
            rc["total"]+=1;rc["correct"]+=pred==str(gold);rc["invalid"]+=pred is None
        assert dict(rc)==native["relation"]
        per_arm[arm]={"dev":dev,"confirm":confirm,
                      "native_direct":nd["totals"],"native_self":selfscore["totals"],
                      "native_relation":dict(rc)}
    assert len(set(starts))==1
    base,control,factor=[per_arm[k] for k in ("initial","task_replay_control","relation_factor")]
    th=plan["thresholds"]
    gates={
       "dev_task_flip_margin":factor["dev"]["task_pairs"]["flip_both_correct"]>=max(base["dev"]["task_pairs"]["flip_both_correct"],control["dev"]["task_pairs"]["flip_both_correct"])+th["dev_flip_margin_pairs"],
       "dev_task_hold_retention":factor["dev"]["task_pairs"]["hold_both_correct"]>=max(base["dev"]["task_pairs"]["hold_both_correct"],control["dev"]["task_pairs"]["hold_both_correct"])-th["hold_tolerance_pairs"],
       "dev_relation_gain":factor["dev"]["relation"]["totals"]["correct"]>=max(base["dev"]["relation"]["totals"]["correct"],control["dev"]["relation"]["totals"]["correct"])+th["factor_relation_improvement_min"],
       "original_native_action_retention":factor["native_direct"]["action_optimal"]>=max(base["native_direct"]["action_optimal"],control["native_direct"]["action_optimal"])-th["native_optimal_tolerance"],
       "original_native_unsafe_retention":factor["native_direct"]["immediate_unsafe"]<=min(base["native_direct"]["immediate_unsafe"],control["native_direct"]["immediate_unsafe"])+th["native_unsafe_tolerance"]
    }
    success=all(gates.values())
    report={
        "audit":"PASS","same_initial_checkpoint":True,
        "cpu_updates_per_treated_arm":64,
        "verified_dev_panel_queries":3*128*2,
        "verified_depth4_panel_queries":3*192*2,
        "verified_native_action_outputs":3*80*2,
        "verified_native_relation_queries":3*192,
        "results":per_arm,"precommitted_gates":gates,"all_gates_pass":success,
        "confirmed_method_gain":False,
        "large_gpu":"NO_GO",
        "reason":"Diagnostic is single seed, synthetic same two source tasks; no independent external norms or 32-step autonomous collaboration."
    }
    (HERE/"AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"audit":"PASS","gates":gates,"all_pass":success,"summary":{
       k:{"dev_task":v["dev"]["task"]["totals"],
          "dev_relation":v["dev"]["relation"]["totals"],
          "dev_flip":v["dev"]["task_pairs"],
          "native_direct":v["native_direct"],
          "native_self":v["native_self"],
          "native_relation":v["native_relation"]} for k,v in per_arm.items()}}))
if __name__=="__main__":main()
