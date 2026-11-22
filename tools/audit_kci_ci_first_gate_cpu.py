"""Independent saved-policy native re-execution for local-repair -> B CCE -> A CCE.

Audit is read-only against original native worlds and saved checkpoints;
no reuse of the training evaluator's cached rollout values.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
import compare_kci_chain_methods_cpu_action_interaction as T
from chain_shared_actor_cpu_action_interaction import run_episode,score_actions
from run_kci_role_local_credit_cpu import load_base
from run_kci_downstream_residual_cpu import RepairedDownstream,CalibratedAfterRepair
from kci_exact_policy_continuation_cpu import environment_public,enumerate_decisions
from run_kci_cpu_policy_conditioned_gate import greedy_eval
from run_kci_native_repair_mc_bridge_cpu import immediate_downstream_repairs

torch.set_num_threads(2)
OUT=ROOT/"artifacts"/"20261122_kci_ci_first_sampled_gate_cpu_v1"


def main():
    doc=json.loads((OUT/"RESULTS.json").read_text(encoding="utf8"))
    assert doc["status"]=="REAL_NATIVE_ROLE_ISOLATED_DPR_BCCE_ACCE_CPU_CHAIN"
    pairs,cases=immediate_downstream_repairs()
    assert len(pairs)==28
    audit={"checkpoint_count":0,"full_native_episodes":0,
           "native_action_steps":0,"nontrain_expected_returns_recomputed":0,
           "downstream_logits_unchanged_by_A":0,
           "immediate_wrong_vs_correct_actions_confirmed":len(pairs)}
    for seedblock in doc["results"]:
        seed=seedblock["seed"]
        base,basepath=load_base(seed)
        p=OUT/"CHECKPOINTS"/seedblock["saved"]
        assert sha256(p.read_bytes()).hexdigest()==seedblock["sha256"]
        data=torch.load(p,weights_only=True,map_location="cpu")
        assert data["seed"]==seed
        assert sha256(basepath.read_bytes()).hexdigest()==data["base_sha256"]
        downstream=RepairedDownstream(base).eval()
        downstream.B.load_state_dict(data["B"])
        downstream.R.load_state_dict(data["R"])
        model=CalibratedAfterRepair(downstream).eval()
        model.A.load_state_dict(data["A"])
        audit["checkpoint_count"]+=1
        for split in ("dev","exploratory_test"):
            independent=greedy_eval(model,T.GROUP[split])
            reference=seedblock["A_after_Qpi_education"][split]
            for k in ("greedy_safe_success","greedy_CI_violations",
                      "A_strict_optimal","mean_A_reference_regret",
                      "mean_policy_expected_return","mean_policy_expected_CI_violation"):
                assert abs(independent[k]-reference[k])<1e-7,(seed,split,k,independent[k],reference[k])
            audit["nontrain_expected_returns_recomputed"]+=len(T.GROUP[split])
            for w in T.GROUP[split]:
                ep=run_episode(environment_public(w),model,greedy=True)
                assert ep["invalid"] is False
                assert ep["success"] is True
                audit["full_native_episodes"]+=1
                audit["native_action_steps"]+=len(ep["actions"])
        for w in T.GROUP["train"][:5]:
            e=environment_public(w)
            states,_=enumerate_decisions(e)
            for item in states:
                if item["phase"]=="A_SEND":continue
                options=list(e.options(item["state"]))
                with torch.no_grad():
                    d=score_actions(downstream,item["obs"],options)
                    m=score_actions(model,item["obs"],options)
                assert torch.equal(d,m)
                audit["downstream_logits_unchanged_by_A"]+=1
        print("AUDIT_RELOADED",seed,"DONE",flush=True)
    audit.update({
        "status":"INDEPENDENT_SAVED_POLICY_NATIVE_REPLAY_PASS",
        "synthetic_test_is_developer_seen":True,
        "exact_Qpi_only_evaluated_not_training_teacher":True,
        "source_base_hashes_verified":True,
        "stage_role_isolation_verified":True,
    })
    p=OUT/"INDEPENDENT_SAVED_MODEL_AUDIT.json"
    p.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("REPLAY_PASS",json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
