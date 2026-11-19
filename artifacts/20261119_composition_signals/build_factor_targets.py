"""Construct verifiable factor-specific targets for native KCI action learning.

No model, no API. Separates relation evidence from permission and task
sufficiency, without changing the original native task oracle.
"""
from __future__ import annotations
import hashlib,json,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/"artifacts"/"20261117_relation_bridge"),str(ROOT/"code")]
from experiments.relation_solution_data import targets
from kci.dynamic_tasks import canonical
from bridge_data import relation_fact, conditional_target
import build_signals
S=HERE/"pilot_01"
def lines(p):return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
    output=S/"factor_targets.jsonl"
    if output.exists():raise FileExistsError("Refuse overwrite of audited factors")
    qs={x["id"]:x for x in lines(S/"queries.jsonl")}
    worlds=lines(S/"worlds.jsonl")
    assert len(worlds)==128 and len(qs)==1536
    records=[]
    for root in worlds:
        group=root["base_group_id"];side=root["side"];world=root["world"]
        native=targets(world)
        assert len(native)==5
        for si,case in enumerate(native):
            obs=case["observation"]
            for ci,c in enumerate(case["candidates"]):
                rid=f"{group}:{side}:{si}:{ci}"
                q=qs[rid]
                assert q["action"]==c["action"] and q["role"]==obs["role"]
                relation=relation_fact(obs,c["action"])
                permission=bool(obs["contract"]["permission"])
                full_required=bool(obs["contract"]["binding_required"])
                label=conditional_target(obs,c["action"],relation)
                # The auxiliary decomposition MUST reproduce the native oracle exactly.
                assert label==c["safe_completion"]==q["label"],rid
                assert relation in (0,1)
                records.append({
                    "id":rid,"base_group_id":group,"split":q["split"],
                    "role":q["role"],"action":q["action"],
                    "relation_new_protected_association":relation,
                    "permission":permission,"full_result_required":full_required,
                    "safe_completion":label,
                    "source":"native_received_ledger_and_finite_task_oracle",
                    "evidence":"relation_fact independently composed with conditional_target; checked against native exact evaluator",
                    "source_id":q["proof"]["state_id"],
                    "training_ready":False
                })
    assert len(records)==len(qs) and len({x["id"] for x in records})==len(qs)
    output.write_text("".join(canonical(r)+"\n" for r in records),encoding="utf-8")
    factor_counts=Counter((x["split"],x["role"],x["relation_new_protected_association"],
                           x["permission"],x["full_result_required"],x["safe_completion"]) for x in records)
    rep={"total":len(records),"train":sum(x["split"]=="train" for x in records),
         "dev":sum(x["split"]=="dev" for x in records),
         "relation_positive":sum(x["relation_new_protected_association"] for x in records),
         "agreement_with_native_task_oracle":"1536/1536",
         "factor_cells":{str(k):v for k,v in sorted(factor_counts.items())},
         "training_ready":False,"independent_business_generalization":False,
         "unknown_norms_present":False,"external_norm_reviewed":False,
         "data_sha256":sha(output)}
    (S/"factor_quality.json").write_text(json.dumps(rep,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in rep.items() if k!="factor_cells"},ensure_ascii=False))
    return rep
if __name__=="__main__":build()
