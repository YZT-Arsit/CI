"""Regression tests for outcome-blind CCE supervision discovery."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from kci_cce_discovery import (
    context_from_source, decision_record, collect_episode, replay,
    select_states, compare_branches, audit_teachers, validate_protocol,
)


@pytest.fixture
def context():
    from kci.v4_phase2_workflows import make_workflows
    from dataclasses import asdict
    wf = next(w for w in make_workflows() if w.split == "train")
    return context_from_source({"native_root_spec": asdict(wf), "condition": "risky"})


class FirstPolicy:
    model_forwards = 0
    tokenized_input_tokens = 0
    cached_score_lookups = 0

    def get_distribution(self, obs):
        # A fixture for native replay, explicitly not learned-policy evidence.
        actions = sorted(obs["legal_actions"])
        act = next((a for a in actions if a != "ABORT"), "ABORT")
        return {"distribution": {a: float(a == act) for a in actions}}


def test_context_identity_includes_norm_not_only_native_state(context):
    a = decision_record(context, [])
    authorized = {**context, "authorized": True}
    b = decision_record(authorized, [])
    assert a["native_state_hash"] == b["native_state_hash"]
    assert a["decision_id"] != b["decision_id"]


def test_visited_state_has_replayable_real_policy_prefix(context):
    episode = collect_episode(context, FirstPolicy(), 71, "fixture")
    assert episode["visits"]
    for visit in episode["visits"]:
        env, state = replay(visit["context"], visit["prefix"])
        assert env.observation(state) == visit["observation"]
        assert visit["selected_action"] in visit["action_probabilities"]
        assert visit["action_probabilities"][visit["selected_action"]] == 1
    assert episode["terminal"]["done"]


def test_selection_ignores_future_outcomes_and_keeps_visit_denominator(context):
    visits = collect_episode(context, FirstPolicy(), 71, "fixture")["visits"]
    before = select_states(visits, per_stratum=1, max_states=3)
    altered = copy.deepcopy(visits)
    for x in altered:
        x["CCE_correct"] = False
        x["terminal_success"] = True
    after = select_states(altered[::-1], per_stratum=1, max_states=3)
    assert before == after
    assert before["visited_occurrences"] == len(visits)
    assert all(len(s["fork_actions"]) == 2 for s in before["states"])


def outcome(seed, *, risk=False, success=False, unknown=False):
    return {"seed": seed, "ci_violation": risk, "ci_safe_success": success,
            "unknown": unknown, "continuation_steps": 10}


def test_no_observed_risk_is_not_a_safety_certificate():
    a = [outcome(i, success=True) for i in range(8)]
    b = [outcome(i) for i in range(8)]
    r = compare_branches(a, b, alpha=0.01, risk_ceiling=0.10)
    assert r["mean_return_preference"] == 0
    assert r["cce_preference"] is None
    assert min(r["risk_upper_bounds"]) > 0.1
    assert r["status"] == "ABSTAIN_RISK_NOT_BOUNDED"


def test_actual_risk_contrast_accepted_but_unknown_and_seed_mismatch_block():
    a = [outcome(i, success=True) for i in range(16)]
    b = [outcome(i, risk=True) for i in range(16)]
    r = compare_branches(a, b, alpha=0.01, risk_ceiling=0.10)
    assert r["cce_preference"] == 0 and r["dimension"] == "risk"
    b[0]["unknown"] = True
    assert compare_branches(a, b, alpha=0.01, risk_ceiling=0.1)["cce_preference"] is None
    b[0]["seed"] = 800
    with pytest.raises(ValueError, match="seed"):
        compare_branches(a, b, alpha=0.01, risk_ceiling=0.1)


def test_success_needs_risk_budget_and_cost_is_not_unconditional_utility():
    a = [outcome(i, success=True) for i in range(256)]
    b = [outcome(i) for i in range(256)]
    r = compare_branches(a, b, alpha=0.01, risk_ceiling=0.1)
    assert r["cce_preference"] == 0 and r["dimension"] == "safe_completion"
    both = compare_branches(a, a, alpha=0.01, risk_ceiling=0.1)
    assert both["cce_preference"] is None


def test_unresolved_observed_risk_cannot_be_overridden_by_success():
    a = [outcome(i, success=True) for i in range(256)]
    b = [outcome(i) for i in range(256)]
    a[0] = outcome(0, risk=True)
    result = compare_branches(a, b, alpha=0.01, risk_ceiling=0.1)
    assert max(result["risk_upper_bounds"]) < 0.1
    assert result["cce_preference"] is None
    assert result["status"] == "ABSTAIN_UNRESOLVED_RISK"


def test_identical_teachers_are_negative_control_not_rejection():
    rows = [{"teacher_actions": {k: "a" for k in ("action_ce_witness", "terminal_K8", "CCE_confirmed")}}] * 7
    a = audit_teachers(rows)
    assert a["same_labels"] == 7 and a["classification"] == "SIGNAL_IDENTITY_CONTROL"
    assert not a["supports_CCE_falsification"]


def test_protocol_rejects_test_selection_fake_reference_and_seed_reuse():
    p = {"collection_seeds": [1], "discovery_seeds": [2], "validation_seeds": [3],
         "collector_sha256": "a", "reference_sha256": "b", "selection_split": "train",
         "risk_ceiling": 0.1, "alpha": 0.05}
    validate_protocol(p)
    for patch in ({"selection_split": "dev"}, {"reference_sha256": "a"}, {"validation_seeds": [2]}):
        with pytest.raises(ValueError):
            validate_protocol({**p, **patch})


def test_real_native_end_to_end_freeze_visit_fork_and_missing_budget(context, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import kci_cce_discovery_run as runner
    from kci_cce_discovery import branch_rollout

    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"native_root_spec": context["spec"], "condition": "risky"}) + "\n", encoding="utf8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"split": "train", "role_observation": decision_record(context, [])["observation"],
        "teacher_actions": {k: "ABORT" for k in ("action_ce_witness", "terminal_K8", "CCE_confirmed")}}) + "\n", encoding="utf8")
    for name in ("base", "collector", "reference"):
        path = tmp_path / name
        path.mkdir()
        files = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json") if name == "base" else ("adapter_model.safetensors", "adapter_config.json")
        for filename in files:
            (path / filename).write_text(name, encoding="utf8")
    args = SimpleNamespace(study=tmp_path / "study", source=source, base=tmp_path / "base",
         collector=tmp_path / "collector", reference=tmp_path / "reference", teachers=[labels],
         legacy_labels=labels, k=2, max_states=3, smoke=False)
    with pytest.raises(ValueError, match="budget cannot confirm"):
        runner.freeze(args)
    assert not args.study.exists()
    args.smoke = True
    runner.freeze(args)
    monkeypatch.setattr(runner, "load_policy", lambda *a: FirstPolicy())
    runner.collect(args)
    pool = json.loads((args.study / "STATE_POOL.json").read_text())
    assert pool["states"]
    bad = copy.deepcopy(pool["states"][0])
    bad["context"]["authorized"] = True
    with pytest.raises(ValueError, match="identity"):
        branch_rollout(bad, bad["fork_actions"][0], FirstPolicy(), 1, "fixture")
    args.policy = "collector"
    def occupied(*a):
        raise RuntimeError("GPU occupied")
    monkeypatch.setattr(runner, "load_policy", occupied)
    with pytest.raises(RuntimeError, match="occupied"):
        runner.fork(args)
    assert not (args.study / "collector_FORK_START.json").exists()
    monkeypatch.setattr(runner, "load_policy", lambda *a: FirstPolicy())
    for name in ("collector", "reference"):
        args.policy = name
        runner.fork(args)
    path = args.study / "reference_FORKS.jsonl"
    saved = path.read_text()
    path.write_text("\n".join(saved.splitlines()[:-1]), encoding="utf8")
    with pytest.raises(ValueError, match="incomplete"):
        runner.analyze(args)
    assert not (args.study / "ANALYSIS.json").exists()
    path.write_text(saved, encoding="utf8")
    runner.analyze(args)
    report = json.loads((args.study / "ANALYSIS.json").read_text())
    assert report["confirmed_labels"] == 0
    assert not report["stage_C_signal_gate_passed"]
    assert json.loads((args.study / "CONFIRMED_LABELS.json").read_text()) == []
    with pytest.raises(FileExistsError):
        runner.freeze(args)
