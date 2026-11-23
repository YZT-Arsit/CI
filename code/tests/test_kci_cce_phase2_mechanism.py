"""End-to-end source-backed CCE Phase2 K8 mechanism integrity tests."""
from pathlib import Path
from collections import Counter,defaultdict
import json,sys
ROOT=Path(__file__).resolve().parents[2]
D=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1"
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_cce_phase2_preference import compare,SEEDS
def r(name):return json.loads((D/name).read_text(encoding="utf8"))
def jl(name):return [json.loads(s) for s in (D/name).read_text(encoding="utf8").splitlines() if s.strip()]
def test_frozen_upstream_A_two_legal_real_deliveries():
    forks=jl("FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl")
    assert len(forks)==14
    assert len({x["root"] for x in forks})==7
    assert {x["condition"] for x in forks}=={"risky","authorized"}
    assert sum(x["deadline_class"]=="tight" for x in forks)==8
    for f in forks:
        a=f["A_valid_candidates"]
        assert set(a)=={"A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"}
        assert all(x["A_verdict"]=="VALID" and x["knowledge_delta"] for x in a.values())
        assert len({x["post_A_native_hash"] for x in a.values()})==2
        assert f["current_A_both_actions_native_CI_legal"]
        assert f["split"]=="train"
        assert f["root"].startswith("P2") and f["root"] not in ("P211","P212")
def test_actual_4B_traces_and_CCE_abstention_not_Qstar():
    xs=jl("POLICY_SAMPLED_4B_K8/SAMPLED_ACTUAL_FROZEN_4B_TRAJECTORIES.jsonl")
    meta=r("POLICY_SAMPLED_4B_K8/SAMPLE_SUMMARY.json")
    assert len(xs)==meta["n_actual_native_team_trajectories"]==224
    assert meta["n_forked_upstream_states"]==14
    assert meta["n_per_action_policy_sample_seeds"]==8
    assert meta["actual_4B_Qpi_sample_evidence_not_native_Qstar"]
    assert meta["n_native_violations"]==sum(x["ci_violation"] for x in xs)==11
    assert meta["n_safe_task_successes"]==sum(x["ci_safe_success"] for x in xs)==53
    assert meta["n_safe_aborts"]==sum(x["aborted"] for x in xs)==142
    assert meta["n_deadline_misses"]==sum(x["deadline_missed"] for x in xs)==18
    assert not any(x["unknown_or_unfinished"] for x in xs)
    assert sum(len(x["native_transition_trace"]) for x in xs)==1550
    counter=Counter((x["cell_id"],x["A_first_action"]) for x in xs)
    assert len(counter)==28 and set(counter.values())=={8}
    for x in xs:
        assert x["A_immediate_verdict"]=="VALID" and not x["A_immediate_CI_violation"]
        assert x["native_transition_trace"][0]["native_action"]==x["A_first_action"]
        assert x["native_transition_trace"][0]["native_verdict"]=="VALID"
        assert x["model_action_count"]==sum(x["downstream_model_decision_roles"].values())
        assert x["seed"] in SEEDS
        assert x["is_actual_frozen_4B_Qpi_sample_not_oracle_Qstar"]
def test_actual_comparison_preference_and_ci_first_uncertainty():
    a=jl("CCE_ANALYSIS_K8/CCE_PAIR_PREFERENCES.jsonl")
    s=r("CCE_ANALYSIS_K8/CCE_MECHANISM_SUMMARY.json")
    assert len(a)==14
    assert s["CCE_accepted"]==0 and s["CCE_abstained"]==14
    assert s["terminal_accepted_no_uncertainty_gate"]==9
    assert s["terminal_given_direction_when_CCE_abstained"]==9
    assert s["no_Qstar_labels_used_for_CCE_or_terminal_preference"]
    reasons=Counter(x["CCE_abstain_reason"] for x in a)
    assert reasons=={"unresolved_nonzero_future_CI_risk_difference":6,
                     "uncertain_success_difference":3,
                     "insufficient_pairwise_safe_success_for_cost":3,
                     "cost_tie_or_uncertain":2}
    assert all(x["CCE_preference"] is None for x in a)
    assert all(x["A_current_action_verdicts"]=={
         "A_SEND_PACKAGE_A":"VALID","A_SEND_PACKAGE_B":"VALID"} for x in a)
    assert all(x["Qstar_late_join_unused_for_labeling"]["NOT_Qpi"] for x in a)
def test_terminal_propagation_loses_direction_on_seed_omission():
    t=r("TERMINAL_PROPAGATION_STABILITY.json")
    s=t["summary"]
    assert s["terminal_directions_K8"]==9
    assert s["n_jackknife_changed_including_tie"]==5
    assert s["n_half_split_changed_including_tie"]==5
    assert s["CI_risk_unresolved_but_scalar_direction_chosen"]==6
    assert s["scalar_wrong_with_available_Qstar_count"]==0
    assert sum(x["terminal_all_K8"] is not None and not x["jackknife_preserves_full"]
               for x in t["cells"])==5
def test_contract_frozen_before_sampling_matches_checkpoint():
    x=r("CCE_COMPARISON_CONTRACT.json")
    actual=r("POLICY_SAMPLED_4B_K8/SAMPLE_SUMMARY.json")
    assert x["frozen_policy_adapter_SHA256"]==actual["model_adapter_sha256"]
    assert x["K_per_A"]==actual["n_per_action_policy_sample_seeds"]==8
    assert x["paired_sample_seeds"]==actual["sample_seeds"]
    assert x["frozen_policy_not_trained_or_modified_in_cell"]
    assert x["no_dev_or_final_test_root_use"]
