"""TRAIN-only, policy-visited CCE discovery; no model or torch import required.

This changes the *study*, not historical frozen labels or the environment.
Same-prefix continuations estimate Q of a named policy, never oracle Q*.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
from math import comb, isfinite, log, sqrt
from pathlib import Path
from random import Random
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from kci.v4_phase2_workflows import Phase2Env, Workflow, WorkStep


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf8")).hexdigest()


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def context_from_source(row):
    spec = row["native_root_spec"]
    if spec["split"] != "train" or row["condition"] not in ("risky", "authorized"):
        raise ValueError("requires an explicit TRAIN risky/authorized source")
    return {"spec": spec, "variant": "risky", "authorized": row["condition"] == "authorized",
            "reorder": False, "package_swap": None}


def replay(context, prefix):
    obj = dict(context["spec"])
    if obj["split"] != "train":
        raise ValueError("TRAIN only")
    for phase in ("pre", "post"):
        obj[phase] = tuple(WorkStep(**{**s, "depends": tuple(s["depends"])}) for s in obj[phase])
    env = Phase2Env(Workflow(**obj), **{k: context[k] for k in
                   ("variant", "authorized", "reorder", "package_swap")})
    state = env.initial()
    for act in prefix:
        if state.done or act not in env.options(state):
            raise ValueError("invalid policy prefix")
        state, verdict = env.transition(state, act)
        if verdict["verdict"] not in ("VALID", "VIOLATION"):
            raise ValueError("UNKNOWN policy prefix")
    return env, state


def decision_record(context, prefix):
    env, state = replay(context, prefix)
    obs = env.observation(state)
    native = asdict(state)
    return {"decision_id": digest({"context": context, "state": native, "observation": obs}),
            "root": context["spec"]["id"], "authorized": context["authorized"],
            "context": context, "prefix": list(prefix), "stage": state.stage,
            "native_state_hash": digest(native), "observation": obs,
            "observation_hash": digest(obs)}


def sample(policy, obs, rng):
    probs = policy.get_distribution(obs)["distribution"]
    if (set(probs) != set(obs["legal_actions"]) or
        any(not isfinite(p) or p < 0 for p in probs.values()) or
        abs(sum(probs.values()) - 1) > 1e-5):
        raise ValueError("invalid policy distribution or removed ABORT")
    draw = rng.random()
    cumulative = 0.
    for action in sorted(probs):
        cumulative += probs[action]
        if draw < cumulative:
            return action, probs, draw
    return sorted(probs)[-1], probs, draw


def terminal(state, steps, unknown):
    return {"done": state.done, "ci_violation": state.violation,
            "ci_safe_success": bool(state.done and state.success and not state.violation),
            "aborted": state.aborted, "deadline_missed": state.deadline_missed,
            "unknown": bool(unknown or not state.done), "continuation_steps": steps,
            "native_total_steps": state.steps, "terminal_state_hash": digest(asdict(state)),
            "recipient_ledger": state.ledger, "business_result": state.business_result}


def collect_episode(context, policy, seed, policy_id, *, max_steps=48):
    env, state = replay(context, [])
    rng, prefix, visits = Random(seed), [], []
    unknown = False
    for _ in range(max_steps):
        if state.done:
            break
        visit = decision_record(context, prefix)
        action, probs, draw = sample(policy, visit["observation"], rng)
        state, verdict = env.transition(state, action)
        visit.update(selected_action=action, action_probabilities=probs,
                     random_draw=draw, verdict=verdict, after_hash=digest(asdict(state)))
        visits.append(visit)
        prefix.append(action)
        if verdict["verdict"] not in ("VALID", "VIOLATION"):
            unknown = True
            break
    return {"root": context["spec"]["id"], "authorized": context["authorized"],
            "seed": seed, "policy_id": policy_id, "initial_state_start": True,
            "visits": visits, "terminal": terminal(state, len(prefix), unknown)}


def select_states(visits, *, per_stratum, max_states):
    """Only current state validity is consulted. No future or teacher filtering."""
    if per_stratum < 1 or max_states < 1:
        raise ValueError("positive state budgets required")
    unique = {}
    for visit in visits:
        record = decision_record(visit["context"], visit["prefix"])
        if record["decision_id"] != visit["decision_id"]:
            raise ValueError("corrupted visited state")
        # Multiple genuine histories can converge. Keep a deterministic witness.
        old = unique.get(record["decision_id"])
        if old is None or tuple(record["prefix"]) < tuple(old["prefix"]):
            unique[record["decision_id"]] = record
    strata = defaultdict(list)
    eligible = 0
    for record in sorted(unique.values(), key=lambda r: r["decision_id"]):
        env, state = replay(record["context"], record["prefix"])
        valid = [a for a in sorted(env.options(state)) if a != "ABORT"
                 and env.transition(state, a)[1]["verdict"] == "VALID"]
        if len(valid) < 2:
            continue
        eligible += 1
        # Predeclared hash selection when more than two current-valid actions exist.
        pair = min(combinations(valid, 2), key=lambda p: digest([record["decision_id"], p]))
        record["fork_actions"] = list(pair)
        strata[(record["root"], record["authorized"], record["stage"])].append(record)
    # Round robin over strata prevents large workflows crowding out smaller ones.
    selected = []
    keys = sorted(strata, key=lambda k: digest(k))
    for rank in range(per_stratum):
        for key in keys:
            if rank < len(strata[key]):
                selected.append(strata[key][rank])
    return {"visited_occurrences": len(visits), "unique_visited_states": len(unique),
            "eligible_unique_states": eligible, "states": selected[:max_states],
            "selection": "hash-stratified TRAIN current-valid non-ABORT pairs; no continuation outcomes"}


def branch_rollout(record, action, policy, seed, policy_id, *, max_steps=48):
    fresh = decision_record(record["context"], record["prefix"])
    if fresh["decision_id"] != record["decision_id"] or action not in record["fork_actions"]:
        raise ValueError("fork state identity mismatch")
    env, state = replay(record["context"], record["prefix"])
    start_steps = state.steps
    rng, events, unknown = Random(seed), [], False
    for index in range(max_steps):
        if state.done:
            break
        obs = env.observation(state)
        if index == 0:
            chosen, probs, draw = action, None, None
        else:
            chosen, probs, draw = sample(policy, obs, rng)
        before = digest(asdict(state))
        state, verdict = env.transition(state, chosen)
        events.append({"action": chosen, "observation_hash": digest(obs),
                       "before_hash": before, "after_hash": digest(asdict(state)),
                       "probabilities": probs, "draw": draw, "verdict": verdict})
        if index == 0 and verdict["verdict"] != "VALID":
            raise ValueError("fork action is not current-valid")
        if verdict["verdict"] not in ("VALID", "VIOLATION"):
            unknown = True
            break
    return {"decision_id": record["decision_id"], "action": action, "seed": seed,
            "policy_id": policy_id, "events": events,
            **terminal(state, state.steps - start_steps, unknown)}


def paired_binary(a, b):
    plus, minus = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
    n = plus + minus
    p = min(1., 2 * sum(comb(n, k) for k in range(max(plus, minus), n + 1)) / 2**n) if n else 1.
    return {"plus": plus, "minus": minus, "p": p,
            "preference": 0 if plus > minus else 1 if minus > plus else None}


def risk_upper(k, n, alpha):
    """Exact one-sided Clopper-Pearson bound, also for zero observed events."""
    if k == n:
        return 1.
    lo, hi = 0., 1.
    for _ in range(64):
        mid = (lo + hi) / 2
        cdf = sum(comb(n, i) * mid**i * (1-mid)**(n-i) for i in range(k+1))
        if cdf > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def compare_branches(a, b, *, alpha, risk_ceiling):
    if not a or len(a) != len(b) or [r["seed"] for r in a] != [r["seed"] for r in b]:
        raise ValueError("paired seed mismatch")
    if len({r["seed"] for r in a}) != len(a):
        raise ValueError("duplicate seed")
    if not 0 < alpha < 1 or not 0 < risk_ceiling < 1:
        raise ValueError("invalid alpha or risk ceiling")
    n = len(a)
    reward = lambda r: -1 if r["ci_violation"] else int(r["ci_safe_success"])
    means = [sum(map(reward, rows))/n for rows in (a, b)]
    risk = paired_binary([not r["ci_violation"] for r in a], [not r["ci_violation"] for r in b])
    success = paired_binary([r["ci_safe_success"] for r in a], [r["ci_safe_success"] for r in b])
    upper = [risk_upper(sum(r["ci_violation"] for r in rows), n, alpha) for rows in (a, b)]
    # Paired scalar differences lie in [-2,2]; this is a mean test, not a sign test.
    radius = sqrt(8 * log(2/alpha) / n)
    diff = means[0] - means[1]
    result = {"n": n, "alpha_per_test": alpha, "risk_test": risk, "success_test": success,
              "value_estimates": [{"n": n,
                  "p_violation": sum(r["ci_violation"] for r in rows) / n,
                  "p_safe_success": sum(r["ci_safe_success"] for r in rows) / n,
                  "mean_cost_given_safe_success": (sum(r["continuation_steps"] for r in rows if r["ci_safe_success"]) /
                      sum(r["ci_safe_success"] for r in rows)) if any(r["ci_safe_success"] for r in rows) else None}
                  for rows in (a, b)],
              "ci_first_point_preference": risk["preference"] if risk["preference"] is not None else success["preference"],
              "risk_upper_bounds": upper, "risk_ceiling": risk_ceiling,
              "mean_returns": means, "mean_return_preference": 0 if diff > 0 else 1 if diff < 0 else None,
              "confidence_same_reward_preference": (0 if diff > 0 else 1) if abs(diff) > radius else None,
              "paired_mean_return_interval": [max(-2.,diff-radius), min(2.,diff+radius)],
              "cce_preference": None, "dimension": None,
              "conditional_costs_descriptive_only": [[r["continuation_steps"] for r in rows if r["ci_safe_success"]] for rows in (a,b)]}
    if any(r["unknown"] for r in a+b):
        result.update(status="ABSTAIN_UNKNOWN", mean_return_preference=None,
                      confidence_same_reward_preference=None, ci_first_point_preference=None)
    elif risk["p"] <= alpha:
        result.update(status="ACCEPT_RISK", cce_preference=risk["preference"], dimension="risk")
    elif risk["plus"] + risk["minus"]:
        result["status"] = "ABSTAIN_UNRESOLVED_RISK"
    elif max(upper) > risk_ceiling:
        result["status"] = "ABSTAIN_RISK_NOT_BOUNDED"
    elif success["p"] <= alpha:
        result.update(status="ACCEPT_SAFE_COMPLETION", cce_preference=success["preference"], dimension="safe_completion")
    else:
        result["status"] = "ABSTAIN_UNCERTAIN_OR_TIED"
    return result


def audit_teachers(rows):
    names = ("action_ce_witness", "terminal_K8", "CCE_confirmed")
    same = sum(len({r["teacher_actions"][k] for k in names}) == 1 for r in rows)
    return {"n_states": len(rows), "same_labels": same,
            "classification": "SIGNAL_IDENTITY_CONTROL" if rows and same == len(rows) else "DIFFERENT_LABELS",
            "supports_CCE_falsification": False,
            "scope": "labels only; objective/schedule/checkpoint provenance audited separately"}


def validate_protocol(p):
    groups = [p[k] for k in ("collection_seeds", "discovery_seeds", "validation_seeds")]
    flat = sum(groups, [])
    if any(not g for g in groups) or len(flat) != len(set(flat)):
        raise ValueError("collection, discovery and validation seeds must be nonempty and disjoint")
    if p["collector_sha256"] == p["reference_sha256"]:
        raise ValueError("reference must be a separately fixed checkpoint")
    if p["selection_split"] != "train":
        raise ValueError("reference and state selection must be TRAIN only")
    if not 0 < p["risk_ceiling"] < 1 or not 0 < p["alpha"] < 1:
        raise ValueError("predeclare risk budget and alpha")
