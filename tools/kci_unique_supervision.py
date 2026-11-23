"""Training admission for CCE evidence, not a new value estimator.

Operate on the fixed discovery runner's cells. Missing evidence is UNKNOWN;
an unseen observation and a rejected baseline label are not useful labels.
"""
from __future__ import annotations

from collections import Counter
from math import isfinite


def conditional_capability(evaluation, policy_sha256):
    """Reuse the frozen executor coverage thresholds for each named checkpoint."""
    if evaluation["model_sha256"] != policy_sha256:
        raise ValueError("capability checkpoint mismatch")
    cal, initial, rare = (evaluation[k] for k in ("calibration", "initial_tasks", "scarce_capabilities"))
    return (cal["n"] > 0 and cal["safe_success"] / cal["n"] >= .9
            and bool(evaluation["per_stage"])
            and all(v["n"] > 0 and v["safe_success"] / v["n"] >= .8
                    for v in evaluation["per_stage"].values())
            and rare["n"] > 0 and rare["safe_success"] == rare["n"]
            and evaluation["legal_RAW_cases"] > 0
            and evaluation["legal_RAW_retained"] / evaluation["legal_RAW_cases"] >= .9
            and cal["unknown"] + initial["unknown"] == 0)


def classify_cell(cell):
    actions = cell["fork_actions"]
    if len(actions) != 2 or len(set(actions)) != 2:
        raise ValueError("two distinct current-valid actions required")
    estimates = cell["estimates"]
    complete = all(set(estimates.get(p, {})) == {"discovery", "validation"}
                   for p in ("collector", "reference"))
    values = [estimates[p][b] for p in ("collector", "reference")
              for b in ("discovery", "validation")] if complete else []
    for blocks in estimates.values():
        for value in blocks.values():
            for key in ("cce_preference", "mean_return_preference", "confidence_same_reward_preference"):
                if value[key] not in (None, 0, 1):
                    raise ValueError("invalid preference index")
    choices = {(v["cce_preference"], v["dimension"]) for v in values}
    confirmed = (complete and len(choices) == 1
                 and values[0]["cce_preference"] is not None
                 and all(v["status"].startswith("ACCEPT_") for v in values))
    chosen = actions[values[0]["cce_preference"]] if confirmed else None
    previous = cell["existing_teacher_actions"]
    # A recorded successful action is not an exhaustive oracle preference set.
    ce_relation = ("unconfirmed" if not confirmed else "uncovered" if not previous
                   else "same_action" if chosen in previous else "different_estimand_or_conflict")
    terminal = estimates.get("collector", {})
    terminal_labels = {b: (None if v["mean_return_preference"] is None
                           else actions[v["mean_return_preference"]])
                       for b, v in terminal.items()}
    controls = {key: {b: (None if v.get(key) is None else actions[v[key]])
                      for b, v in terminal.items()}
                for key in ("confidence_same_reward_preference", "ci_first_point_preference")}
    terminal_same = confirmed and all(terminal_labels.get(b) == chosen
                                      for b in ("discovery", "validation"))
    terminal_gap = confirmed and not terminal_same
    # Different scalar and CI-first objectives are not an estimator advantage.
    candidate = confirmed and ce_relation == "uncovered" and terminal_gap
    return {"decision_id": cell["decision_id"], "root": cell["root"],
            "stage": cell["stage"], "chosen_action": chosen,
            "action_ce_actions": previous, "action_ce_relation": ce_relation,
            "terminal_actions": terminal_labels, "evidence_complete": complete,
            "control_actions": controls, "block_evidence": estimates,
            "cce_confirmed": confirmed, "new_exact_observation": not previous,
            "incremental_vs_recorded_ce": confirmed and ce_relation == "uncovered",
            "terminal_direction_gap": terminal_gap,
            "candidate_unique_useful": candidate,
            "objective_matched_advantage_established": False,
            "reason": ("MISSING_REFERENCE_OR_BLOCK" if not complete else
                       "CCE_ABSTAIN" if not confirmed else
                       "REDUNDANT_WITH_TERMINAL" if terminal_same else
                       "CE_CONFLICT_IS_NOT_CORRECTION" if ce_relation == "different_estimand_or_conflict" else
                       "REDUNDANT_WITH_CE" if ce_relation == "same_action" else
                       "NEW_CI_FIRST_SUPERVISION_REQUIRES_MATCHED_OBJECTIVE_ABLATION")}


def summarize(cells, *, competence, costs, smoke=False):
    if len({c["decision_id"] for c in cells}) != len(cells):
        raise ValueError("duplicate decision comparison")
    records = [classify_cell(c) for c in cells]
    count = lambda field: sum(bool(r[field]) for r in records)
    needed = ("collection", "collector", "reference", "witness_search", "validation")
    # No imputed zeros. A historical search with no cost ledger is unknown.
    units = ("candidate_forwards", "input_tokens", "native_transitions", "wall_seconds")
    complete_cost = all(name in costs and all(
        isinstance(costs[name].get(k), (int, float))
        and not isinstance(costs[name][k], bool)
        and isfinite(costs[name][k]) and costs[name][k] >= 0 for k in units)
        for name in needed)
    measured = {k: sum(v[k] for v in costs.values() if isinstance(v.get(k), (int, float))
                       and not isinstance(v[k], bool) and isfinite(v[k]) and v[k] >= 0)
                for k in units}
    capable = all(competence.get(p, {}).get("passed") is True
                  and competence.get(p, {}).get("conditional_capability_passed") is True
                  for p in ("collector", "reference"))
    blockers = []
    if not records:
        blockers.append("EMPTY_COMPARISON_POOL")
    if not all(r["evidence_complete"] for r in records):
        blockers.append("INCOMPLETE_CONTINUATION_EVIDENCE")
    if not capable:
        blockers.append("REFERENCE_OR_COLLECTOR_CAPABILITY_NOT_PASSED")
    if not complete_cost:
        blockers.append("INCOMPLETE_TOTAL_COST_LEDGER")
    if not count("candidate_unique_useful"):
        blockers.append("NO_NONREDUNDANT_CONFIRMED_SUPERVISION")
    if smoke:
        blockers.append("SMOKE_ONLY")
    ready = not blockers
    useful = count("candidate_unique_useful") if ready else 0
    return {"schema": "kci_unique_useful_supervision_v1", "cells": records,
            "n_comparisons": len(records), "n_roots": len({r["root"] for r in records}),
            "confirmed_cce": count("cce_confirmed"),
            "new_exact_observations": count("new_exact_observation"),
            "incremental_vs_recorded_ce": count("incremental_vs_recorded_ce"),
            "terminal_direction_gaps": count("terminal_direction_gap"),
            "candidate_unique_useful": count("candidate_unique_useful"),
            "unique_useful_supervision_established": useful,
            "reasons": dict(Counter(r["reason"] for r in records)),
            "costs": costs, "known_cost_subtotal": measured,
            "complete_total_cost": complete_cost,
            "cost_per_established_unique_preference": {k: v / useful for k, v in measured.items()} if useful else None,
            "cost_accounting": "each branch estimator charged full common sampling; witness search and independent validation not free",
            "blockers": blockers, "signal_gate_passed": ready,
            "training_admitted": False,
            "decision": "READY_FOR_MATCHED_OBJECTIVE_COMPARISON_PROTOCOL" if ready else "NO_CCE_TRAINING_INCREMENT_NOT_ESTABLISHED",
            "required_next_control": "CI-first point-estimate teacher on identical forks; isolate objective change from confidence filtering",
            "scope": "registered teacher inventory; missing CE label is not proof CE cannot acquire it; no independent-root claim"}


