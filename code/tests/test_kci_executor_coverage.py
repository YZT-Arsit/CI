"""A sampling repair must preserve labels, loss and stage budget."""
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from kci_executor_coverage import coverage_schedule, freeze, stage_cycle


def fixture_rows():
    rows = []
    for stage in set(stage_cycle()):
        actions = ("RAW", "VERIFY", "BROKER") if stage == "release" else ("STEP",)
        for action in actions:
            for n in range(50 if action == "RAW" else 1):
                rows.append({"stage": stage, "root": "R1", "chosen_action": action,
                             "same_observation_sha256": f"{stage}:{action}:{n}", "split": "train"})
    return rows


def test_rare_actions_receive_exposure_without_changing_stage_budget():
    rows = fixture_rows()
    schedule = coverage_schedule(rows, seed=7, steps=160, batch=2)
    assert len(schedule) == 320
    expected = Counter(stage_cycle()[i % len(stage_cycle())] for i in range(320))
    assert Counter(r["stage"] for r in schedule) == expected
    counts = Counter(r["chosen_action"] for r in schedule if r["stage"] == "release")
    assert min(counts.values()) >= 20
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(any(r is old for old in rows) for r in schedule)


def test_seed_only_changes_within_stratum_order_and_no_dev():
    rows = fixture_rows()
    a = coverage_schedule(rows, seed=7, steps=160, batch=2)
    b = coverage_schedule(rows, seed=19, steps=160, batch=2)
    assert Counter((r["stage"], r["chosen_action"]) for r in a) == Counter((r["stage"], r["chosen_action"]) for r in b)
    assert a == coverage_schedule(rows[::-1], seed=7, steps=160, batch=2)
    rows[0]["split"] = "dev"
    with pytest.raises(ValueError, match="TRAIN"):
        coverage_schedule(rows, seed=7, steps=160, batch=2)


def test_missing_stage_or_upstream_A_supervision_fails():
    rows = fixture_rows()
    with pytest.raises(ValueError, match="stages"):
        coverage_schedule([r for r in rows if r["stage"] != "broker"], seed=7, steps=160, batch=2)
    rows.append({**rows[0], "stage": "A_SEND"})
    with pytest.raises(ValueError, match="stages"):
        coverage_schedule(rows, seed=7, steps=160, batch=2)


def test_freeze_replays_teacher_witnesses_and_binds_candidate_budget(tmp_path):
    target = tmp_path / "study"
    result = freeze(target)
    assert result["objective_unchanged"] and result["teacher_labels_unchanged"]
    assert result["source_unique_observations"] == 1358
    assert result["independent_train_roots"] == 7
    assert result["new_schedule_counts"]["VERIFY_PRIVATE_TARGET"] >= 20
    assert result["new_schedule_counts"]["OPEN_PRIVATE_CHANNEL"] >= 20
    assert result["old_schedule_counts"]["VERIFY_PRIVATE_TARGET"] == 1
    assert result["native_success_witnesses_replayed"] == 405
    assert result["calibration_cases"] > 30
    assert result["candidate_models_allowed"] == 1
    assert all("\\" not in p for p in result["input_hashes"])
    assert "\\" not in result["train_file"] and "\\" not in result["baseline_adapter"]
    from kci_executor_coverage_run import protocol, continue_policy
    import json
    panel = json.loads((target / "CALIBRATION_PANEL.json").read_text(encoding="utf8"))
    class AbortPolicy:
        def get_distribution(self, observation):
            assert "teacher_action" not in observation
            return {"distribution": {a: float(a == "ABORT") for a in observation["legal_actions"]}}
    outcome = continue_policy(panel[0]["context"], panel[0]["prefix"], AbortPolicy(), 71)
    assert outcome["aborted"] and not outcome["ci_safe_success"] and not outcome["unknown"]
    assert len(outcome["events"]) == 1
    protocol(target)
    (target / "CALIBRATION_PANEL.json").write_text("[]", encoding="utf8")
    with pytest.raises(ValueError, match="panel changed"):
        protocol(target)
    with pytest.raises(FileExistsError):
        freeze(target)
