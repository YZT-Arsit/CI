"""Saved real 4B CCE post-SFT experiment: end-to-end invariants."""
from pathlib import Path
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_cce_adaptive_protocol import (SEEDS32,ALPHA_CONFIRM,stage_signal,
                                       ACTION_A,ACTION_B)
A=ROOT/"artifacts/20261124_cce_downstream_sft_v1"
B=ROOT/"artifacts/20261124_cce_phase2_post_sft_v1"
C=ROOT/"artifacts/20261124_cce_method_comparison_v1"
def read(p):return json.loads(p.read_text(encoding="utf8"))
def jl(p):return [json.loads(x) for x in p.open(encoding="utf8") if x.strip()]
def test_native_sft_witnesses_and_ABORТ_actions():
    data=jl(A/"SHARED_ROLE_ACTION_SFT_TRAIN_ONLY.jsonl")
    evidence=read(A/"SFT_WITNESS_MANIFEST.json")
    assert evidence["native_full_success_witnesses"]==405
    assert evidence["unique_role_observation_sft_samples"]==len(data)==1358
    assert len({x["same_observation_sha256"] for x in data})==len(data)
    assert not any(r["stage"]=="A_SEND" for r in data)
    assert all(r["ABORT_preserved_in_action_menu"] and "ABORT" in r["all_executable_actions"] for r in data)
    assert set(evidence["observation_stages"])=={"post","A_READ","broker","evidence","release","attest","preflight","pre"}
def test_train_SFT_real_4B_steps_and_one_shared_adapter():
    j=read(A/"shared_native_sft_seed7/TRAINING.json")
    assert j["sampled_sft_steps"]==160 and j["sampled_train_observation_updates"]==320
    assert j["differentiable_candidate_forwards"]==750
    assert j["one_shared_adapter_for_all_roles"]
    assert j["ABORT_preserved_in_every_model_menu"]
    assert j["A_SEND_not_supervised_by_success_witnesses"]
    assert j["DEV_and_final_roots_never_read"]
    assert (A/"shared_native_sft_seed7/adapter/adapter_model.safetensors").stat().st_size>10000000
def test_original_new_K8_exactly_matched_and_risk_tradeoff_preserved():
    j=read(B/"BEFORE_AFTER_MATCHED_K8.json")
    assert j["same_A_states"]==14 and j["exactly_same_A_action_and_sample_seed_per_pair"]
    assert j["before"]["n"]==j["after"]["n"]==224
    assert j["before"]["compliant_business_completed"]==53
    assert j["after"]["compliant_business_completed"]==169
    assert j["before"]["future_CI_violations"]==11
    assert j["after"]["future_CI_violations"]==55
    assert j["before"]["safe_aborts"]==142
    assert j["after"]["safe_aborts"]==0
def test_adaptive_all_7_accepted_CI_first_no_ties_converted_to_labels():
    j=read(B/"POST_SFT_FROZEN_ADAPTIVE/SUMMARY.json")
    pairs=jl(B/"POST_SFT_FROZEN_ADAPTIVE/ADAPTIVE_CELL_DECISIONS.jsonl")
    assert len(pairs)==14
    assert j["n_real_frozen_4B_native_trajectories"]==672
    assert j["confirmed_CCE_preferences"]==7 and j["abstained_CCE_preferences"]==7
    assert j["confirmed_dimensions"]=={"ACCEPT_CI_safety":7}
    assert j["abstention_reasons"]=={"ABSTAIN_EQUAL_COST":7}
    assert j["K_per_action_distribution"]=={"16":7,"32":7}
    for x in pairs:
        y=x["look_history"][-1]["evidence"]["current_look"]
        if x["condition"]=="risky":
            assert y["confirmed"] and y["dimension"]=="CI_safety"
            assert y["paired_sign"]["exact_two_sided_p"]<=ALPHA_CONFIRM
            assert y["block_consistency"]["passed"]
            assert x["confirmed_CCE_preference"]==x["K8_terminal_scalar_propagation"]["chosen"]
        else:
            assert x["condition"]=="authorized"
            assert not y["confirmed"] and x["confirmed_CCE_preference"] is None
def test_cheap_local_closure_covers_all_current_confirmed_CCE_labels():
    r=read(B/"STATIC_LOCAL_CLOSURE_AUDIT.json")
    assert r["CCE_accepted_count"]==7
    assert r["static_heuristic_match_CCE_on_accepted"]==7
    assert r["static_heuristic_unresolved_on_authorized"]==7
    assert r["never_uses_Qstar_label_as_input"]
def test_matched_three_signal_teacher_data_frozen_without_DEV():
    rows=jl(C/"MATCHED_SAME_A_STATE_THREE_TEACHERS_TRAIN_ONLY.jsonl")
    manifest=read(C/"THREE_ARM_PRE_REGISTRATION.json")
    assert len(rows)==7 and {x["root"] for x in rows}=={
        "P201","P202","P203","P205","P206","P207","P208"}
    assert manifest["three_method_supervision_labels_equal_counts"]=={
       "terminal_equals_CCE":7,"ActionCE_equals_CCE":7}
    assert all(x["teacher_actions"]["action_ce_witness"]==
               x["teacher_actions"]["terminal_K8"]==
               x["teacher_actions"]["CCE_confirmed"] for x in rows)
    assert all("ABORT" in x["allowed_native_A_actions"] for x in rows)
    assert manifest["no_DEV_or_final_test_evaluated_or_train_labels"]
