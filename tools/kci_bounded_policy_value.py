"""Finite-environment Qpi bounds from actual frozen-model action distributions.

Unknown model decisions retain their entire probability mass as uncertainty.
This is a white-box finite-model diagnostic, not a black-box CCE trial.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite


CHANCE = {"PRIMARY_CHANCE", "SECOND_CHANCE", "AUTH_CHANCE"}


def normalized(distribution, actions):
    if set(distribution) != set(actions) or not distribution:
        raise ValueError("action menu mismatch")
    if any(not isfinite(v) or v < 0 for v in distribution.values()):
        raise ValueError("invalid model probability")
    total = sum(distribution.values())
    if abs(total - 1) > 1e-5:
        raise ValueError("model probabilities not normalized")
    return {a: p / total for a, p in distribution.items()}


def evaluate(env, start, cache, model_view, state_digest, *, max_states=100000, structural_bounds=False):
    """Return resolved outcome mass and frontier reach mass; no guessed policy."""
    memo, frontier_views, envelopes = {}, {}, {}
    def envelope(s):
        key = state_digest(s.__dict__)
        if key in envelopes:
            return envelopes[key]
        if len(envelopes) >= max_states:
            raise RuntimeError("native envelope state cap exceeded")
        if s.done:
            risk, success = float(s.violation), float(s.success and not s.violation)
            result = [risk, risk, success, success]
        elif s.stage in CHANCE:
            result = [0.] * 4
            for p, _, after in env.chance_outcomes(s):
                bound = envelope(after)
                for i in range(4):
                    result[i] += float(p) * bound[i]
        else:
            bounds = []
            for action in env.options(s):
                after, verdict = env.transition(s, action)
                if verdict["verdict"] not in ("VALID", "VIOLATION"):
                    raise ValueError("native envelope UNKNOWN")
                if not after.done and after.steps <= s.steps:
                    raise ValueError("nonprogressing native envelope")
                bounds.append(envelope(after))
            result = [min(b[0] for b in bounds), max(b[1] for b in bounds),
                      min(b[2] for b in bounds), max(b[3] for b in bounds)]
        envelopes[key] = result
        return result
    def rec(s):
        key = state_digest(s.__dict__)
        if key in memo:
            return memo[key]
        if len(memo) >= max_states:
            raise RuntimeError("finite traversal state cap exceeded")
        if s.done:
            success = bool(s.success and not s.violation)
            out = ([float(s.violation), float(success), s.steps * float(success), 0.,
                    float(s.aborted and not s.violation), float(s.deadline_missed), float(s.steps),
                    float(s.violation), float(s.violation), float(success), float(success)], {})
        elif s.stage in CHANCE:
            children = [(float(p), after) for p, _, after in env.chance_outcomes(s)]
            if abs(sum(p for p, _ in children) - 1) > 1e-10:
                raise ValueError("invalid native chance support")
            out = combine(children)
        else:
            view = model_view(env.actor_observation(s))
            prompt = view["prompt_sha256"]
            if prompt not in cache:
                frontier_views[prompt] = view
                bounds = envelope(s) if structural_bounds else [0., 1., 0., 1.]
                out = ([0., 0., 0., 1., 0., 0., 0.] + bounds, {prompt: 1.})
            else:
                probs = normalized(cache[prompt], view["available_actions"])
                children = []
                for action, prob in probs.items():
                    if prob == 0:
                        continue
                    after, verdict = env.transition(s, action)
                    if verdict["verdict"] not in ("VALID", "VIOLATION"):
                        raise ValueError("native UNKNOWN")
                    if not after.done and after.steps <= s.steps:
                        raise ValueError("nonprogressing native actor transition")
                    children.append((prob, after))
                out = combine(children)
        memo[key] = out
        return out

    def combine(children):
        total, frontier = [0.] * 11, defaultdict(float)
        for weight, child in children:
            values, unresolved = rec(child)
            for i, value in enumerate(values):
                total[i] += weight * value
            for prompt, mass in unresolved.items():
                frontier[prompt] += weight * mass
        return total, dict(frontier)

    values, frontier = rec(start)
    risk, success, cost_success, missing, abort, deadline, elapsed = values[:7]
    return {"p_violation_interval": values[7:9],
            "p_safe_success_interval": values[9:11],
            "unresolved_policy_mass": missing, "resolved_safe_abort_mass": abort,
            "resolved_deadline_mass": deadline,
            "resolved_success_cost_mass": cost_success,
            "expected_cost_given_safe_success": cost_success / success if missing == 0 and success else None,
            "resolved_expected_elapsed": elapsed,
            "finite_policy_fully_evaluated": missing == 0,
            "native_states_visited": len(memo), "native_envelope_states": len(envelopes),
            "structural_bounds": structural_bounds,
            "frontier": frontier}, frontier_views


def compare(a, b, *, tolerance=1e-6):
    """Only disjoint risk bounds certify a direction while policy mass is missing."""
    ra, rb = a["p_violation_interval"], b["p_violation_interval"]
    if ra[1] + tolerance < rb[0]:
        return {"choice": "SEND_A", "dimension": "risk", "status": "BOUNDED_POLICY_RISK_SEPARATION"}
    if rb[1] + tolerance < ra[0]:
        return {"choice": "SEND_B", "dimension": "risk", "status": "BOUNDED_POLICY_RISK_SEPARATION"}
    if not a["finite_policy_fully_evaluated"] or not b["finite_policy_fully_evaluated"]:
        return {"choice": None, "dimension": None, "status": "UNRESOLVED_POLICY_MASS"}
    # Tolerance means numerically indistinguishable, not a certificate of equal risk.
    if ra[0] != rb[0]:
        return {"choice": None, "dimension": None, "status": "RISK_DIFFERENCE_BELOW_NUMERICAL_TOLERANCE"}
    sa, sb = a["p_safe_success_interval"][0], b["p_safe_success_interval"][0]
    if abs(sa - sb) > tolerance:
        return {"choice": "SEND_A" if sa > sb else "SEND_B", "dimension": "safe_success",
                "status": "EXACT_FINITE_POLICY_SUCCESS_SEPARATION"}
    return {"choice": None, "dimension": None, "status": "TIE_OR_COST_NOT_CERTIFIED"}
