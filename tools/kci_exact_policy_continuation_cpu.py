"""Exact CPU continuation evaluator for the existing KCI A->R->B->C actor.

This is DEV-ONLY. It expands real native state transitions, with the
probability of EACH subsequent decision taken from a FROZEN learned actor.
The actor sees only env.observation(state), never Qstar or hidden labels.
No gradients or oracle-optimal suffixes are used to calculate Qpi.
"""
from __future__ import annotations

from dataclasses import asdict
from collections import Counter
from pathlib import Path
from copy import deepcopy
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "tools")]

from kci.diverse_ci_branch_env import digest
from kci.upstream_downstream_chain_pilot_public_contract import ChainPilotEnvPublicContract
from chain_shared_actor_cpu_action_interaction import (
    spec_from_world, valid_decision_candidates, baseline_automatic_action,
    score_actions,
)

DECISION_STAGES = ("A_SEND", "preflight", "release")
IGNORED = {"ABORT", "ABORT_CHAIN"}


def environment_public(row):
    """Only adds an already-public cost contract to upstream observation."""
    return ChainPilotEnvPublicContract(
        spec_from_world(row["root"], row["world"]),
        swap=row["swapped_package_names"],
        source_mode=row["source_mode"],
        safe_surcharge=row["source_cost_surcharge"],
    )


def stage_name(observation):
    return observation.get("stage", observation.get("current_stage", "unknown"))


def terminal_vector(env, st):
    n = st.native
    success = bool(n.done and n.success and not n.violation)
    violation = bool(n.violation)
    reward = (round(1.0 - env.spec.cost_per_step * n.steps, 6)
              if success else (-1.0 if violation else 0.0))
    return reward, float(success), float(violation)


class FrozenPolicyValue:
    """Exact expected return/success/violation for a fixed stochastic actor."""

    def __init__(self, env, frozen_actor):
        self.env = env
        self.actor = frozen_actor.eval()
        self.cache = {}
        self.queries = 0
        self.transitions = 0
        self.policy_action_scores = 0

    def q(self, state, action):
        nxt, audit = self.env.transition(state, action)
        self.transitions += 1
        if audit["verdict"] not in ("VALID", "VIOLATION"):
            raise AssertionError((audit, action))
        return self.value(nxt)

    def value(self, state):
        key = digest(asdict(state))
        if key in self.cache:
            return self.cache[key]
        self.queries += 1
        if state.native.done:
            result = terminal_vector(self.env, state)
        else:
            obs = self.env.observation(state)
            available = list(self.env.options(state))
            decisions = valid_decision_candidates(obs, available)
            if len(decisions) >= 2:
                with torch.no_grad():
                    p = torch.softmax(score_actions(self.actor, obs, decisions), dim=0)
                    probabilities = p.detach().cpu().tolist()
                self.policy_action_scores += 1
                values = [self.q(state, action) for action in decisions]
                result = tuple(sum(float(pi) * v[k] for pi, v in zip(probabilities, values))
                               for k in range(3))
            else:
                action = baseline_automatic_action(self.env, state)
                if action is None:
                    result = (0.0, 0.0, 0.0)
                else:
                    result = self.q(state, action)
        self.cache[key] = result
        return result


def enumerate_decisions(env):
    """Enumerate actual reachable non-abort prefixes, not oracle suffixes."""
    todo = [env.initial()]
    seen = set()
    points = []
    audit = Counter()
    while todo:
        state = todo.pop()
        key = digest(asdict(state))
        if key in seen:
            continue
        seen.add(key)
        if state.native.done:
            continue
        obs = env.observation(state)
        options = list(env.options(state))
        actions = valid_decision_candidates(obs, options)
        phase = stage_name(obs)
        if phase in DECISION_STAGES and len(actions) > 1:
            active = [a for a in actions if a not in IGNORED]
            if len(active) == 2:
                verdicts = {a: env.transition(state, a)[1]["verdict"] for a in active}
                if set(verdicts.values()) <= {"VALID", "VIOLATION"}:
                    points.append({
                        "state": state, "obs": deepcopy(obs), "hash": key,
                        "phase": phase, "role": obs.get("actor", obs.get("role")),
                        "actions": tuple(active), "verdicts": verdicts,
                    })
                    audit["points_" + phase] += 1
        if len(actions) >= 2:
            candidates = [a for a in actions if a not in IGNORED]
        else:
            a = baseline_automatic_action(env, state)
            candidates = [a] if a else []
        for a in candidates:
            nxt, native_audit = env.transition(state, a)
            if native_audit["verdict"] not in ("VALID", "VIOLATION"):
                raise AssertionError((native_audit, a))
            todo.append(nxt)
    audit["all_native_states"] = len(seen)
    return points, dict(audit)


def policy_preference_points(env, actor, points, min_abs_gap=0.004,
                             include_repair=False):
    evaluator = FrozenPolicyValue(env, actor)
    buckets = {"A_SEND": [], "preflight": [], "release": [], "repair": []}
    stats = Counter()
    for p in points:
        a, b = p["actions"]
        av, bv = p["verdicts"][a], p["verdicts"][b]
        q_a = evaluator.q(p["state"], a)[0]
        q_b = evaluator.q(p["state"], b)[0]
        phase = p["phase"]
        if "VIOLATION" in (av, bv):
            if phase == "release" and include_repair:
                good = a if av == "VALID" else b
                bad = b if good == a else a
                buckets["repair"].append((p["obs"], good, bad, abs(q_a-q_b), p["hash"]))
                stats["dpr_verified_pairs"] += 1
            else:
                stats["unsafe_excluded_from_cce"] += 1
        elif abs(q_a-q_b) >= min_abs_gap:
            better, worse = (a, b) if q_a > q_b else (b, a)
            buckets[phase].append((p["obs"], better, worse, abs(q_a-q_b), p["hash"]))
            stats["cce_" + phase] += 1
        else:
            stats["small_gap_" + phase] += 1
    stats["exact_policy_states_evaluated"] = evaluator.queries
    stats["native_transitions"] = evaluator.transitions
    stats["policy_action_scores"] = evaluator.policy_action_scores
    return buckets, dict(stats)
