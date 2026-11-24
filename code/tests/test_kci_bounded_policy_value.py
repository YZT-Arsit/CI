from dataclasses import dataclass, replace
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from kci_bounded_policy_value import evaluate, compare, normalized


@dataclass
class State:
    stage: str = "actor"
    done: bool = False
    success: bool = False
    violation: bool = False
    aborted: bool = False
    deadline_missed: bool = False
    steps: int = 0


class Env:
    def actor_observation(self, s):
        return {"prompt_sha256": s.stage, "available_actions": ["safe", "bad", "stop"]}

    def transition(self, s, action):
        return replace(s, done=True, steps=s.steps+1, success=action == "safe",
                       violation=action == "bad", aborted=action == "stop"), {
                           "verdict": "VIOLATION" if action == "bad" else "VALID"}

    def chance_outcomes(self, s):
        return [(0.25, "bad", replace(s, done=True, violation=True, steps=1)),
                (0.75, "continue", replace(s, stage="actor", steps=1))]


def run(state, cache):
    return evaluate(Env(), state, cache, lambda x: x, lambda x: json.dumps(x, sort_keys=True))[0]


def test_unknown_model_mass_never_becomes_safe_success():
    r = run(State(), {})
    assert r["p_violation_interval"] == [0, 1]
    assert r["p_safe_success_interval"] == [0, 1]
    assert r["unresolved_policy_mass"] == 1
    assert compare(r, r)["choice"] is None


def test_finite_model_value_integrates_actual_probabilities_and_chance():
    r = run(State(stage="PRIMARY_CHANCE"), {"actor": {"safe": .6, "bad": .2, "stop": .2}})
    assert r["p_violation_interval"] == pytest.approx([.4, .4])
    assert r["p_safe_success_interval"] == pytest.approx([.45, .45])
    assert r["resolved_safe_abort_mass"] == pytest.approx(.15)
    assert r["expected_cost_given_safe_success"] == 2
    assert r["finite_policy_fully_evaluated"]


def test_bounds_cover_every_possible_unknown_policy():
    bounded = run(State(stage="PRIMARY_CHANCE"), {})
    for action in ("safe", "bad", "stop"):
        exact = run(State(stage="PRIMARY_CHANCE"), {"actor": {a: float(a == action) for a in ("safe", "bad", "stop")}})
        for metric in ("p_violation_interval", "p_safe_success_interval"):
            assert bounded[metric][0] <= exact[metric][0] <= bounded[metric][1]


def test_unresolved_risk_blocks_task_gain_and_near_equal_risk_not_tie():
    a = run(State(), {"actor": {"safe": 1, "bad": 0, "stop": 0}})
    b = run(State(), {})
    assert compare(a, b)["choice"] is None
    b = run(State(), {"actor": {"safe": 0, "bad": 1e-8, "stop": 1-1e-8}})
    assert compare(a, b)["status"] == "RISK_DIFFERENCE_BELOW_NUMERICAL_TOLERANCE"
    b = run(State(), {"actor": {"safe": 0, "bad": .5, "stop": .5}})
    assert compare(a, b)["choice"] == "SEND_A"


def test_invalid_distribution_rejected():
    for d in ({"safe": 1}, {"safe": -1, "bad": 1, "stop": 1},
              {"safe": .9, "bad": .9, "stop": 0}, {"safe": float("nan"), "bad": 0, "stop": 0}):
        with pytest.raises(ValueError):
            normalized(d, ["safe", "bad", "stop"])


def test_structural_bounds_can_certify_no_risk_without_imputing_model_policy():
    class NoLeakEnv(Env):
        def options(self, s):
            return ["safe", "stop"]
        def actor_observation(self, s):
            return {"prompt_sha256": s.stage, "available_actions": self.options(s)}
    r, _ = evaluate(NoLeakEnv(), State(), {}, lambda x: x,
                    lambda x: json.dumps(x, sort_keys=True), structural_bounds=True)
    assert r["unresolved_policy_mass"] == 1
    assert r["p_violation_interval"] == [0, 0]
    assert r["p_safe_success_interval"] == [0, 1]
    assert not r["finite_policy_fully_evaluated"]
