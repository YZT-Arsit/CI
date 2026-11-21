r"""Frozen six-checkpoint exact native Qstar vs Qpi comparison: no model updates.

Exact Qpi uses current frozen SharedHornActor at EVERY future critical action
of the real KCI A->R->B->C simulator; future action-value labels never replace
the frozen policy. Each immediately legal A branch is executed from the SAME
verified real source-tool observation. Enumerate all reachable downstream
paths and probability-weight terminal rewards; preserve native evidence.

This script only READS original 28 worlds and 6 CPU checkpoint weights.
Writes results to a NEW, exclusively named artifact directory.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import json,sys,time,math
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_exact_frozen_qpi_cpu import (
    GROUP,ROWS,KINDS,SEEDS,environment,prepare_A,ExactQpi,
    load_frozen_policy,qstar_A_from_original)
BASE=ROOT/"artifacts"/"20261121_policy_gap_pilot_01"
OUT=BASE/"SIX_SAVED_CHECKPOINTS_QSTAR_VS_EXACT_QPI"
if OUT.exists():raise FileExistsError(OUT)
BASE.mkdir(exist_ok=True)
OUT.mkdir()
qstar_reference={key:qstar_A_from_original(row,verify=True)
                 for key,row in ROWS.items()}
assert len(qstar_reference)==28
checkpoint_meta=[]
reports=[]
summary=defaultdict(Counter)
with (OUT/"ALL_SIX_FROZEN_QPI_EVIDENCE.jsonl").open("x",encoding="utf-8",buffering=1) as f:
    for method in KINDS:
        for seed in SEEDS:
            model,meta=load_frozen_policy(method,seed)
            checkpoint_meta.append(meta)
            start=time.perf_counter()
            for split in ("train","dev","exploratory_test"):
                for row in GROUP[split]:
                    env=environment(row)
                    source=prepare_A(env)
                    estimate=ExactQpi(env,model,meta["checkpoint_sha256"]).evaluate_A_arms(source)
                    star=qstar_reference[row["world"]]
                    gap_star=round(star["PACKAGE_A"]-star["PACKAGE_B"],9)
                    gap_pi=estimate["Qpi_A_minus_B"]
                    is_strict=abs(gap_star)>1e-8
                    # A tied Qstar makes a "conflict" undefined, not a model failure.
                    conflict=bool(is_strict and gap_star*gap_pi<-1e-9)
                    if is_strict: summary[method,seed]["strict_worlds"]+=1
                    else: summary[method,seed]["ties"]+=1
                    summary[method,seed]["worlds"]+=1
                    summary[method,seed]["conflict"]+=int(conflict)
                    summary[method,seed]["native_states"]+=estimate["cost"]["native_states"]
                    summary[method,seed]["frozen_actor_calls"]+=estimate["cost"]["unique_frozen_actor_softmax_calls"]
                    summary[method,seed]["native_edges"]+=estimate["cost"]["actual_native_transition_edges"]
                    re={
                        "world":row["world"],"split":split,
                        "root":row["root"],"graph_family":row["graph"],
                        "source_business":row["business"],"method":method,"seed":seed,
                        "checkpoint":meta,
                        "native_Qstar":star,
                        "Qstar_A_minus_B":gap_star,
                        "frozen_exact_Qpi":estimate,
                        "strict_non_tie":is_strict,
                        "Qstar_Qpi_strict_preference_conflict":conflict,
                        "Qpi_value_computed_from_actual_frozen_model_future_policy":True,
                        "native_Qstar_used_only_after_model_value_for_audit":True,
                    }
                    reports.append({
                        "world":row["world"],"split":split,"kind":method,"seed":seed,
                        "Qstar_gap":gap_star,"Qpi_gap":gap_pi,
                        "conflict":conflict,
                        "native_states":estimate["cost"]["native_states"],
                        "softmax_calls":estimate["cost"]["unique_frozen_actor_softmax_calls"]
                    })
                    f.write(json.dumps(re,ensure_ascii=False,separators=(",",":"),sort_keys=True)+"\n")
                    if row["world"]=="DG19_M01__tradeoff__swap0":
                        print("DG19",method,seed,"Qstar",gap_star,"Qpi",gap_pi,
                              "diff",round(gap_pi-gap_star,6),flush=True)
            print("FINISHED",method,seed,
                  dict(summary[method,seed]),
                  "wall_seconds",round(time.perf_counter()-start,2),flush=True)
    f.flush()
assert all(summary[m,s]["strict_worlds"]==24 and summary[m,s]["ties"]==4
           for m in KINDS for s in SEEDS)
assert all(summary[m,s]["conflict"]==10 for m in KINDS for s in SEEDS),summary
source_hash=sha256((OUT/"ALL_SIX_FROZEN_QPI_EVIDENCE.jsonl").read_bytes()).hexdigest()
stats={
    "status":"ALL_SIX_SAVED_CPU_CHECKPOINTS_EXACT_QPI_REPRODUCED",
    "worlds":28,"source_dataset_modified":False,
    "checkpoint_count":len(checkpoint_meta),
    "checkpoint_identities":checkpoint_meta,
    "strict_Qstar_worlds_per_checkpoint":24,
    "true_ties_per_checkpoint":4,
    "Qstar_Qpi_conflicts_per_checkpoint":{f"{m}_seed{s}":summary[m,s]["conflict"]
       for m in KINDS for s in SEEDS},
    "per_checkpoint":{f"{m}_seed{s}":dict(summary[m,s]) for m in KINDS for s in SEEDS},
    "total_native_states":sum(x["native_states"] for x in reports),
    "total_native_edges":sum(summary[m,s]["native_edges"] for m in KINDS for s in SEEDS),
    "total_policy_softmax_calls":sum(x["softmax_calls"] for x in reports),
    "raw_evidence_sha256":source_hash,
    "saved_checkpoints_frozen":True,
    "all_future_actions_enumerated_not_sampled":True,
    "native_Qstar_is_not_future_model_policy":True,
    "original_data_labels_and_tests_unmodified":True,
    "not_independent_real_world_benchmark":True,
}
(OUT/"SUMMARY.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("SUMMARY",json.dumps({k:v for k,v in stats.items()
       if k not in ("checkpoint_identities","per_checkpoint")},ensure_ascii=False),flush=True)
