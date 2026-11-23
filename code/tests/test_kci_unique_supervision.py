from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from kci_unique_supervision import classify_cell, conditional_capability, summarize


def cell(previous=(), terminal=0):
    value = {"cce_preference": 0, "dimension": "risk", "status": "ACCEPT_RISK",
             "mean_return_preference": terminal, "confidence_same_reward_preference": None}
    return {"decision_id": "s", "root": "train1", "stage": "A_SEND",
            "fork_actions": ["A", "B"], "existing_teacher_actions": list(previous),
            "estimates": {p: {b: deepcopy(value) for b in ("discovery", "validation")}
                          for p in ("collector", "reference")}}


def complete_inputs():
    competence = {p: {"passed": True, "conditional_capability_passed": True}
                  for p in ("collector", "reference")}
    costs = {k: {"candidate_forwards": 10, "input_tokens": 100,
                 "native_transitions": 20, "wall_seconds": 1.}
             for k in ("collection", "collector", "reference", "witness_search", "validation")}
    return {"competence": competence, "costs": costs}


def test_new_observation_same_terminal_label_is_not_unique_supervision():
    r = summarize([cell()], **complete_inputs())
    assert r["incremental_vs_recorded_ce"] == 1
    assert r["candidate_unique_useful"] == 0
    assert not r["signal_gate_passed"]
    assert r["cost_per_established_unique_preference"] is None


def test_genuine_new_ci_first_direction_only_opens_separate_comparison():
    r = summarize([cell(terminal=1)], **complete_inputs())
    assert r["unique_useful_supervision_established"] == 1
    assert r["signal_gate_passed"]
    assert not r["training_admitted"]
    assert not r["cells"][0]["objective_matched_advantage_established"]
    assert r["cost_per_established_unique_preference"]["candidate_forwards"] == 50


def test_disagreement_with_success_witness_is_not_certified_correction():
    r = classify_cell(cell(previous=["B"], terminal=1))
    assert r["action_ce_relation"] == "different_estimand_or_conflict"
    assert not r["candidate_unique_useful"]
    assert not classify_cell(cell(previous=["A", "B"], terminal=1))["candidate_unique_useful"]


@pytest.mark.parametrize("failure", ["missing_reference", "validation_disagrees", "unknown",
                                      "missing_cost", "bad_cost", "weak_capability", "smoke"])
def test_missing_or_invalid_evidence_cannot_open_gate(failure):
    c, kwargs = cell(terminal=1), complete_inputs()
    if failure == "missing_reference":
        del c["estimates"]["reference"]
    elif failure == "validation_disagrees":
        c["estimates"]["reference"]["validation"]["cce_preference"] = 1
    elif failure == "unknown":
        c["estimates"]["collector"]["validation"]["status"] = "ABSTAIN_UNKNOWN"
    elif failure == "missing_cost":
        del kwargs["costs"]["witness_search"]
    elif failure == "bad_cost":
        kwargs["costs"]["validation"]["wall_seconds"] = float("nan")
    elif failure == "weak_capability":
        kwargs["competence"]["reference"]["conditional_capability_passed"] = False
    else:
        kwargs["smoke"] = True
    assert not summarize([c], **kwargs)["signal_gate_passed"]


def test_empty_and_duplicate_pools_do_not_create_evidence():
    assert not summarize([], **complete_inputs())["signal_gate_passed"]
    with pytest.raises(ValueError, match="duplicate"):
        summarize([cell(), cell()], **complete_inputs())


def test_conditional_capability_checks_checkpoint_and_legal_raw():
    e = {"model_sha256": "a", "calibration": {"n": 10, "safe_success": 10, "unknown": 0},
         "initial_tasks": {"unknown": 0}, "scarce_capabilities": {"n": 4, "safe_success": 4},
         "per_stage": {"release": {"n": 10, "safe_success": 10}},
         "legal_RAW_cases": 10, "legal_RAW_retained": 10}
    assert conditional_capability(e, "a")
    with pytest.raises(ValueError, match="checkpoint"):
        conditional_capability(e, "b")
    e["legal_RAW_retained"] = 8
    assert not conditional_capability(e, "a")


def test_retired_trainer_exits_before_loading_model_or_inputs(tmp_path):
    trainer = Path(__file__).resolve().parents[2] / "tools/kci_cce_three_arm_A_train.py"
    cmd = [sys.executable, str(trainer), "--base", "missing", "--shared-sft-adapter", "missing",
           "--train", "missing", "--pre-reg", "missing", "--arm", "CCE", "--output", str(tmp_path / "out")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 2
    assert "same-label seven-state study retired" in result.stderr
    assert not (tmp_path / "out").exists()
