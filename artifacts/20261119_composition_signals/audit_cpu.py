"""Independent read-only result auditor for native KCI CPU matched study.

No new model calls. Refuses partial studies; never changes generated outputs.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json,re,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
S=HERE/"cpu_matched_01"
def sha(path):return sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding="utf-8"))
def rows(path):return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def parse(text,alph,time):
    m=re.findall(r"ANSWER:\s*(["+alph+r"])\b",text)
    return m[0] if len(m)==1 and 0<time<9.5 else None
def audit():
    plan=load(S/"PLAN.json");sealed=load(S/"SEALED.json");overall=load(S/"RESULTS.json")
    assert sealed["complete"] and sealed["plan_hash"]==sha(S/"PLAN.json")
    assert sealed["results_sha256"]==sha(S/"RESULTS.json")
    assert sealed["training_code_hash"]==sha(HERE/"run_cpu.py")
    for rel,h in sealed["files"].items():assert sha(S/rel)==h,rel
    assert set(overall["arms"])=={"initial","paired","shuffled"}
    panel=rows(S/"dev_panel.jsonl")
    assert len(panel)==128 and len({r["id"] for r in panel})==128
    paired={}
    for x in panel: paired.setdefault(x["pair_group"],[]).append(x)
    assert len(paired)==64 and all(len(x)==2 for x in paired.values())
    comp=ROOT/"artifacts"/"20261117_composition_repair"
    queries=rows(comp/"queries.jsonl")
    states=rows(comp/"states.jsonl")
    labels=rows(comp/"labels.jsonl")
    assert (len(queries),len(states),len(labels))==(192,80,192)
    label_map={l["id"]:l for l in labels}
    original_model_sha=[]
    for arm in ("initial","paired","shuffled"):
        folder=S/"study"/arm
        res=overall["arms"][arm]
        output=rows(folder/"dev_outputs.jsonl")
        assert len(output)==128
        got={r["id"]:r for r in output}
        assert set(got)=={r["id"] for r in panel}
        count=Counter();byrole=defaultdict(Counter)
        for x in panel:
            value=got[x["id"]]["output"]
            assert value==parse(got[x["id"]]["raw"],"01",got[x["id"]]["seconds_batch"])
            count["individual_total"]+=1
            count["individual_correct"]+=value==str(x["label"])
            count["unknown"]+=value is None
            byrole[x["role"]]["individual_total"]+=1
            byrole[x["role"]]["individual_correct"]+=value==str(x["label"])
        for group,items in paired.items():
            category=items[0]["pair_label"]
            good=all(got[r["id"]]["output"]==str(r["label"]) for r in items)
            count[category+"_pairs"]+=1
            count[category+"_both_correct"]+=int(good)
            role=items[0]["role"]
            byrole[role][category+"_pairs"]+=1
            byrole[role][category+"_both_correct"]+=int(good)
        assert dict(count)==res["dev"]["summary"],arm
        assert {k:dict(v) for k,v in byrole.items()}==res["dev"]["by_role"],arm
        nq=rows(folder/"native"/"direct_queries.jsonl")
        na=rows(folder/"native"/"direct_actions.jsonl")
        assert [x["id"] for x in nq]==[x["id"] for x in queries]
        assert [x["id"] for x in na]==[x["id"] for x in states]
        ns=Counter()
        native_role=defaultdict(Counter)
        for x,r in zip(queries,nq):
            l=label_map[x["id"]]
            value=parse(r["raw"],"01",r["seconds_batch"])
            correct=value==str(l["truth_target"])
            ns["query_total"]+=1;ns["query_correct"]+=correct
            role=l["case"]["observation"]["role"]
            native_role[role]["query_total"]+=1;native_role[role]["query_correct"]+=correct
        state_case={x["state_id"]:x["case"] for x in labels}
        for x,record in zip(states,na):
            case=state_case[x["id"]]
            val=parse(record["raw"],"ABC",record["seconds_batch"])
            valid=val is not None and ord(val)-65<len(x["actions"])
            selected=x["actions"][ord(val)-65] if valid else None
            assert valid==record["valid"] and selected==record["selected"]
            candidates={c["action"]:c for c in case["candidates"]}
            best={case["candidates"][i]["action"] for i in case["optimal_indices"]}
            success=valid and selected in best
            unsafe=valid and not candidates[selected]["immediate_safe"]
            role=case["observation"]["role"]
            ns["action_total"]+=1;ns["action_optimal"]+=success;ns["immediate_unsafe"]+=unsafe;ns["invalid_actions"]+=not valid
            native_role[role]["action_total"]+=1;native_role[role]["action_optimal"]+=success;native_role[role]["immediate_unsafe"]+=unsafe
        assert dict(ns)==res["native"]["summary"],arm
        assert {k:dict(v) for k,v in native_role.items()}==res["native"]["by_role"]
        identity=load(folder/"IDENTITY.json")
        assert identity["before"]==identity["after"]
        original_model_sha.append(identity["initial"])
        if arm!="initial":
            training=rows(folder/"training.jsonl")
            schedule=rows(S/(arm+"_steps.jsonl"))
            assert len(training)==len(schedule)==32
            assert [r["query_ids"] for r in training]==[r["query_ids"] for r in schedule]
            assert [r["optimizer_step"] for r in training]==list(range(1,33))
            assert all(0<r["seconds"]<30 for r in training)
            assert all(r["query_tokens"]>0 and r["actor_tokens"]>0 for r in training)
            cp=load(folder/"CHECKPOINT_SEALED.json")
            assert cp["steps"]==32 and cp["initial_checkpoint"]==plan["model_start_path"]
            for name,expected in cp["adapter_hashes"].items():assert sha(folder/"adapter"/name)==expected
    assert len(set(original_model_sha))==1, "same initial model not preserved"
    report={"audit":"PASS","complete_arms":3,"checked_dev_queries":384,
            "checked_native_queries":576,"checked_native_actions":240,
            "total_cpu_updates":64,"all_arms_same_start":True,
            "padded_floPs_exactly_equal":False,"external_norm_validation":False,
            "official_kci_method_gain_established":False}
    (S/"INDEPENDENT_AUDIT.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))
if __name__=="__main__":audit()
