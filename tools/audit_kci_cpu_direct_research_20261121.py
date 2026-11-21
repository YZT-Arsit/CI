"""Independent saved-checkpoint native replay of the 2026-11-21 KCI CPU gate.

Loads model weights from disk, reconstructs all training/held-out observations
from original native source cases, and checks full trajectories, expected policy
returns, action-stage isolation and source hashes against frozen result JSON.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
import compare_kci_chain_methods_cpu_action_interaction as T
from chain_shared_actor_cpu_action_interaction import (
    SharedHornActor, score_actions, run_episode,
)
from kci_exact_policy_continuation_cpu import (
    environment_public,enumerate_decisions,FrozenPolicyValue,
)
from run_kci_downstream_residual_cpu import (
    RepairedDownstream,CalibratedAfterRepair,
)
from run_kci_role_local_credit_cpu import load_base
from run_kci_cpu_policy_conditioned_gate import greedy_eval

torch.set_num_threads(2)
OUTPUT=ROOT/"artifacts"/"20261121_kci_downstream_residual_cpu_v1"
SRC=OUTPUT/"RESULTS.json"


def verify():
    payload=json.loads(SRC.read_text(encoding="utf8"))
    counts={"checkpoints":0,"native_episodes":0,"decision_sites_checked":0,
            "recomputed_expected_value_worlds":0,"native_actions_replayed":0,
            "A_value_order_comparisons":0,"Qpi_train_disagree_after_downstream":0,
            "Qpi_train_strict_compared":0}
    details=[]
    for seedblock in payload["results"]:
        seed=seedblock["seed"]
        base,base_path=load_base(seed)
        repaired=RepairedDownstream(base)
        for pair in seedblock["trained_A"]:
            method=pair["method"]
            ckpath=OUTPUT/"CHECKPOINTS"/pair["checkpoint_name"]
            ckhash=hashlib.sha256(ckpath.read_bytes()).hexdigest()
            assert ckhash==pair["sha256"],(method,seed,"checkpoint hash mismatch")
            saved=torch.load(ckpath,map_location="cpu",weights_only=True)
            assert saved["method"]==method and saved["seed"]==seed
            assert hashlib.sha256(base_path.read_bytes()).hexdigest()==saved["source_hash"]
            repaired.B.load_state_dict(saved["B"])
            repaired.R.load_state_dict(saved["R"])
            actor=CalibratedAfterRepair(repaired)
            actor.A.load_state_dict(saved["A"])
            actor.eval()
            counts["checkpoints"]+=1
            # Recompute TRAIN-only action-ranking alignment independently.
            if method == "A_Qpi_after_downstream":
                for train_case in T.GROUP["train"]:
                    ee = environment_public(train_case)
                    ss, aa = ee.transition(ee.initial(), "A_READ_VERIFIED_PACKAGES")
                    assert aa["verdict"] == "VALID"
                    counterfactual = FrozenPolicyValue(ee, repaired)
                    qpi_gap = (
                        counterfactual.q(ss, "A_SEND_PACKAGE_A")[0] -
                        counterfactual.q(ss, "A_SEND_PACKAGE_B")[0])
                    target = train_case["upstream_A_values_given_exact_B_continuation"]
                    qstar_gap = target["PACKAGE_A"] - target["PACKAGE_B"]
                    if abs(qstar_gap) > 1e-8:
                        counts["Qpi_train_strict_compared"] += 1
                        counts["Qpi_train_disagree_after_downstream"] += int(
                            qpi_gap * qstar_gap < 0)

            # Rebuild original native worlds; test every real decision role.
            for split in ("train","dev","exploratory_test"):
                for row in T.GROUP[split]:
                    env=environment_public(row)
                    pts,stats=enumerate_decisions(env)
                    for point in pts:
                        options=list(env.options(point["state"]))
                        with torch.no_grad():
                            core=score_actions(base,point["obs"],options)
                            modified_B=score_actions(repaired,point["obs"],options)
                            modified_A=score_actions(actor,point["obs"],options)
                        if point["phase"]=="A_SEND":
                            # Downstream repair CANNOT modify an upstream decision.
                            assert torch.equal(core,modified_B)
                        else:
                            # A credit CANNOT change any non-A logits.
                            assert torch.equal(modified_B,modified_A)
                        counts["decision_sites_checked"]+=1

                    if split=="train":
                        continue
                    ep=run_episode(env,actor,greedy=True)
                    counts["native_episodes"]+=1
                    counts["native_actions_replayed"]+=len(ep["actions"])
                    value=FrozenPolicyValue(env,actor).value(env.initial())
                    counts["recomputed_expected_value_worlds"]+=1
                    expected=next(x for x in pair["eval"][split]["per_world"]
                                  if x["world"]==row["world"])
                    assert bool(ep["success"])==expected["success"]
                    assert bool(ep["violation"])==expected["ci_violation"]
                    assert abs(ep["return"]-expected["greedy_return"])<1e-8
                    assert abs(value[0]-expected["policy_expected_return"])<1e-6
                    assert abs(value[1]-expected["policy_expected_safe_success"])<1e-6
                    assert abs(value[2]-expected["policy_expected_CI_violation"])<1e-6
                    for item in ep["actions"]:
                        assert item["verdict"] in ("VALID","VIOLATION")
                    oracle=row["upstream_A_values_given_exact_B_continuation"]
                    recorded_action=expected["greedy_A_action"]
                    regret=max(oracle.values())-oracle[recorded_action]
                    assert abs(regret-expected["A_reference_native_regret"])<1e-8
                    counts["A_value_order_comparisons"]+=int(abs(oracle["PACKAGE_A"]-oracle["PACKAGE_B"])>1e-8)
            details.append({"method":method,"seed":seed,"checkpoint_sha256":ckhash,
                            "audit":"PASS"})
    payload_out={
        "status":"INDEPENDENT_SAVED_CPU_ACTOR_NATIVE_REPLAY_PASS",
        "counts":counts,
        "all_recomputed_from_same_original_cases":True,
        "A_training_cannot_modify_downstream_logits":True,
        "downstream_repair_cannot_modify_A_logits":True,
        "source_hashes_verified":True,
        "test_is_developer_seen_not_blind":True,
        "no_claim_of_natural_language_generalization":True,
        "per_checkpoint":details,
    }
    out=OUTPUT/"INDEPENDENT_SAVED_MODEL_REPLAY_AUDIT.json"
    out.write_text(json.dumps(payload_out,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("REPLAY_PASS",json.dumps(counts),flush=True)
    print("AUDIT_FILE",out,flush=True)


if __name__=="__main__":
    verify()
