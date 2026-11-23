"""Freeze -> collect actual TRAIN visits -> fork -> analyze. Never trains models.

Each command writes new files exclusively. Reference competence failures are
reported, not repaired by searching checkpoints against held-out outcomes.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json
import subprocess
import time

from kci_cce_discovery import (
    ROOT, digest, file_sha, read_jsonl, context_from_source, collect_episode,
    select_states, branch_rollout, compare_branches, audit_teachers, validate_protocol,
)


def write(path, value):
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append(f, value):
    f.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    f.flush()


def counters(policy):
    return {"candidate_forwards": policy.model_forwards,
            "input_tokens": policy.tokenized_input_tokens,
            "cache_hits": policy.cached_score_lookups}


def verified_protocol(study):
    p = json.loads((study / "PROTOCOL.json").read_text(encoding="utf8"))
    validate_protocol(p)
    for path, expected in p["input_hashes"].items():
        if file_sha(path) != expected:
            raise ValueError("frozen input changed: " + path)
    return p


def load_policy(p, name):
    usage = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    if any(int(v.strip()) > 512 for v in usage.splitlines()):
        raise RuntimeError("GPU occupied; no model loaded and no process interrupted")
    from kci_cce_phase2_frozen4b_rollout import Frozen4B
    return Frozen4B(p["base"], p[name + "_adapter"])


def freeze(args):
    if args.study.exists():
        raise FileExistsError(args.study)
    if args.k < 1 or args.max_states < 1:
        raise ValueError("positive sampling budgets required")
    alpha_per_test = .05 / (args.max_states * 2 * 2 * 5)
    minimum_binary_p = 2. ** (1 - args.k)
    if minimum_binary_p > alpha_per_test and not args.smoke:
        raise ValueError("budget cannot confirm even perfectly separated binary outcomes; increase K or explicitly use --smoke")
    rows = read_jsonl(args.source)
    contexts = {digest(context_from_source(r)): context_from_source(r) for r in rows}
    inputs = [args.source, *args.teachers, args.legacy_labels,
              Path(__file__), ROOT / "tools/kci_cce_discovery.py",
              ROOT / "tools/kci_cce_phase2_frozen4b_rollout.py",
              ROOT / "code/kci/v4_phase2_workflows.py", ROOT / "code/kci/v4_phase2_data.py",
              ROOT / "code/kci/agent_rl/hf_policy.py"]
    for adapter in (args.collector, args.reference):
        inputs += [adapter / "adapter_model.safetensors", adapter / "adapter_config.json"]
    base_files = sorted(args.base.glob("*.safetensors"))
    if not base_files:
        raise ValueError("base has no model weights")
    inputs += base_files + [args.base / "config.json", args.base / "tokenizer.json", args.base / "tokenizer_config.json"]
    p = {"version": "cce_policy_visit_discovery_v1", "scope": "TRAIN diagnostic, not a method superiority experiment",
         "plumbing_smoke_only": args.smoke,
         "power_check": {"minimum_two_sided_binary_p": minimum_binary_p,
                         "per_test_alpha": alpha_per_test,
                         "binary_confirmation_possible": minimum_binary_p <= alpha_per_test},
         "base": str(args.base.resolve()), "collector_adapter": str(args.collector.resolve()),
         "reference_adapter": str(args.reference.resolve()),
         "collector_sha256": file_sha(args.collector / "adapter_model.safetensors"),
         "reference_sha256": file_sha(args.reference / "adapter_model.safetensors"),
         "reference_selection": "fixed before sampling; historical independent teacher checkpoint; never rank by DEV/TEST or CCE agreement",
         "selection_split": "train", "source": str(args.source.resolve()),
         "teachers": [str(t.resolve()) for t in args.teachers],
         "legacy_labels": str(args.legacy_labels.resolve()),
         "contexts": [contexts[k] for k in sorted(contexts)],
         "collection_seeds": [810011, 810013],
         "discovery_seeds": list(range(820000, 820000 + args.k)),
         "validation_seeds": list(range(830000, 830000 + args.k)),
         "per_stratum": 1, "max_states": args.max_states, "alpha": .05, "risk_ceiling": .10,
         "reference_competence_min_safe_completion": .5,
         "reference_competence_max_unknown": 0,
         "inference_scope": "conditional Monte Carlo estimates for two named Qpi; reference is not ground truth",
         "statistics": "one fixed look per block; Bonferroni max_states x 2 policies x 2 blocks x 5 tests; no optional stopping",
         "five_tests": ["paired_binary_risk", "paired_binary_safe_success", "risk_upper_A", "risk_upper_B", "bounded_paired_scalar_mean"],
         "cost": "conditional safe-success cost descriptive only; never an unconditional utility certificate",
         "seed_coupling": "common random numbers per action pair; not identical exogenous noise after paths diverge",
         "selection": "full initial-state policy visits, current validity only, hash-stratified before any fork outcomes",
         "abort": "always available to policy; two non-ABORT forks used to study lawful-action credit",
         "full_evaluation_gate": "requires competence + independently confirmed new observation supervision; current generator shortcut still blocks generalization claims",
         "C_comparison": {
             "main_arms": ["adapted_PrivAct_descendant_mean", "full_role_authorized_history_CE", "CCE"],
             "auxiliary": ["single_trajectory_terminal_return", "confidence_filter_same_scalar_reward"],
             "shared_signal_study": "same state pool, same fork samples, common Action CE optimizer; explicitly teacher-acquisition comparison",
             "end_to_end_study": "separate pre-registration needed for online occupancy and different optimizers; no claim of faithful PrivAct from Action CE",
             "budget": "charge collection + witness search + forks + validation + training forwards/tokens; matched ceilings and coverage-cost curves, not only equal updates",
             "capacity": "same base/init/context window/role-visible history/actions/optimizer schedule for shared-signal arms",
             "evaluation": "fresh root split frozen before training; root-clustered compliant success, violation, UNKNOWN, deadline, safe abort and avoidable protection only with a feasible-witness denominator",
             "equal_labels": "retain as negative control; identical targets do not falsify CCE",
             "prohibited": "DEV/TEST reference selection; CCE-wins state selection; fabricated weak baseline labels; counting seeds as independent roots"},
         "input_hashes": {str(f.resolve()): file_sha(f) for f in inputs}}
    validate_protocol(p)
    args.study.mkdir(parents=True)
    write(args.study / "PROTOCOL.json", p)
    write(args.study / "LEGACY_SIGNAL_AUDIT.json", audit_teachers(read_jsonl(args.legacy_labels)))
    print("FROZEN", digest(p), "contexts", len(contexts), flush=True)


def collect(args):
    p = verified_protocol(args.study)
    path = args.study / "VISITED_EPISODES.jsonl"
    if path.exists():
        raise FileExistsError(path)
    policy = load_policy(p, "collector")
    start, episodes = time.monotonic(), []
    with path.open("x", encoding="utf8") as f:
        for context in p["contexts"]:
            for seed in p["collection_seeds"]:
                row = collect_episode(context, policy, seed, p["collector_sha256"])
                append(f, row)
                episodes.append(row)
                print("VISIT", row["root"], row["authorized"], seed, row["terminal"]["ci_safe_success"], flush=True)
    pool = select_states([v for e in episodes for v in e["visits"]],
                         per_stratum=p["per_stratum"], max_states=p["max_states"])
    pool["source_episode_sha256"] = file_sha(path)
    pool["protocol_sha256"] = file_sha(args.study / "PROTOCOL.json")
    write(args.study / "STATE_POOL.json", pool)
    write(args.study / "COLLECTION_COST.json", {**counters(policy), "wall_seconds": time.monotonic()-start,
          "model_load_excluded": True, "episodes": len(episodes),
          "native_transitions": sum(len(e["visits"]) for e in episodes)})


def fork(args):
    p = verified_protocol(args.study)
    pool_path = args.study / "STATE_POOL.json"
    pool = json.loads(pool_path.read_text(encoding="utf8"))
    if pool["source_episode_sha256"] != file_sha(args.study / "VISITED_EPISODES.jsonl") or pool["protocol_sha256"] != file_sha(args.study / "PROTOCOL.json"):
        raise ValueError("state pool provenance mismatch")
    marker = args.study / (args.policy + "_FORK_START.json")
    if marker.exists():
        raise FileExistsError(marker)
    # Occupancy/model-load failures must not consume the exclusive start marker.
    policy = load_policy(p, args.policy)
    write(marker, {"pool_sha256": file_sha(pool_path)})
    start = time.monotonic()
    competence = []
    with (args.study / (args.policy + "_COMPETENCE.jsonl")).open("x", encoding="utf8") as f:
        for context in p["contexts"]:
            for seed in p["collection_seeds"]:
                episode = collect_episode(context, policy, seed, p[args.policy + "_sha256"])
                append(f, episode)
                competence.append(episode)
    competence_cost = counters(policy)
    outcomes = []
    with (args.study / (args.policy + "_FORKS.jsonl")).open("x", encoding="utf8") as f:
        for record in pool["states"]:
            for block in ("discovery", "validation"):
                for seed in p[block + "_seeds"]:
                    for action in record["fork_actions"]:
                        row = branch_rollout(record, action, policy, seed, p[args.policy + "_sha256"])
                        row["block"] = block
                        append(f, row)
                        outcomes.append(row)
            print("FORK", args.policy, record["root"], record["stage"], record["decision_id"][:10], flush=True)
    write(args.study / (args.policy + "_COST.json"), {
        **counters(policy), "competence_cumulative": competence_cost,
        "branch_only": {k: v - competence_cost[k] for k, v in counters(policy).items()},
        "wall_seconds": time.monotonic() - start, "model_load_excluded": True,
        "branch_episodes": len(outcomes), "branch_native_transitions": sum(len(r["events"]) for r in outcomes),
        "competence_native_transitions": sum(len(e["visits"]) for e in competence),
        "sample_accounting": "mean, confidence-same-reward and CCE each charged full common branch sample budget; validation is additional, not free"})


def analyze(args):
    p = verified_protocol(args.study)
    pool_path = args.study / "STATE_POOL.json"
    pool = json.loads(pool_path.read_text(encoding="utf8"))
    teacher = {}
    for path in p["teachers"]:
        for row in read_jsonl(path):
            obs = row.get("native_role_observation", row.get("role_observation"))
            action = row.get("chosen_action", row.get("teacher_actions", {}).get("action_ce_witness"))
            if obs is None or action is None or row["split"] != "train":
                raise ValueError("requires TRAIN teacher observation and action")
            teacher.setdefault(digest(obs), set()).add(action)
    competence, evidence = {}, {}
    alpha = p["alpha"] / (p["max_states"] * 2 * 2 * 5)
    for name in ("collector", "reference"):
        start = json.loads((args.study / (name + "_FORK_START.json")).read_text(encoding="utf8"))
        if start["pool_sha256"] != file_sha(pool_path):
            raise ValueError("fork pool changed after freeze")
        episodes = read_jsonl(args.study / (name + "_COMPETENCE.jsonl"))
        expected = {(digest(c), s) for c in p["contexts"] for s in p["collection_seeds"]}
        actual = {(digest(e["visits"][0]["context"]), e["seed"]) for e in episodes}
        if actual != expected or len(episodes) != len(expected):
            raise ValueError("incomplete competence episodes")
        n = len(episodes)
        successes = sum(e["terminal"]["ci_safe_success"] for e in episodes)
        unknown = sum(e["terminal"]["unknown"] for e in episodes)
        competence[name] = {"n": n, "safe_success": successes, "unknown": unknown,
            "passed": successes/n >= p["reference_competence_min_safe_completion"] and unknown <= p["reference_competence_max_unknown"],
            "per_root": {r: {"n": sum(e["root"] == r for e in episodes), "success": sum(e["root"] == r and e["terminal"]["ci_safe_success"] for e in episodes)} for r in sorted({e["root"] for e in episodes})}}
        rows = read_jsonl(args.study / (name + "_FORKS.jsonl"))
        expected_n = len(pool["states"]) * 2 * (len(p["discovery_seeds"]) + len(p["validation_seeds"]))
        if len(rows) != expected_n or any(r["policy_id"] != p[name + "_sha256"] for r in rows):
            raise ValueError("incomplete/mixed policy branch data")
        for state in pool["states"]:
            for block in ("discovery", "validation"):
                pair = [[r for r in rows if r["decision_id"] == state["decision_id"] and r["block"] == block and r["action"] == a] for a in state["fork_actions"]]
                if any([r["seed"] for r in arm] != p[block + "_seeds"] for arm in pair):
                    raise ValueError("missing, duplicated or out-of-order budget")
                evidence[(name, state["decision_id"], block)] = compare_branches(*pair, alpha=alpha, risk_ceiling=p["risk_ceiling"])
    cells, labels = [], []
    for state in pool["states"]:
        estimates = {name: {block: evidence[(name, state["decision_id"], block)] for block in ("discovery", "validation")} for name in ("collector", "reference")}
        results = [v for blocks in estimates.values() for v in blocks.values()]
        choices = {(v["cce_preference"], v["dimension"]) for v in results}
        robust = len(choices) == 1 and results[0]["cce_preference"] is not None
        previous = sorted(teacher.get(state["observation_hash"], set()))
        cell = {"decision_id": state["decision_id"], "root": state["root"], "stage": state["stage"],
                "fork_actions": state["fork_actions"], "existing_teacher_actions": previous,
                "new_exact_observation": not previous, "estimates": estimates,
                "independently_confirmed_across_named_policies": robust}
        if robust:
            choice = state["fork_actions"][results[0]["cce_preference"]]
            cell["chosen_action"] = choice
            cell["existing_teacher_disagreement_not_proven_teacher_error"] = bool(previous and choice not in previous)
            labels.append({"decision_id": state["decision_id"], "observation": state["observation"],
                           "chosen_action": choice, "evidence": cell})
        cells.append(cell)
    ready = not p["plumbing_smoke_only"] and all(c["passed"] for c in competence.values()) and any(c["new_exact_observation"] and c["independently_confirmed_across_named_policies"] for c in cells)
    report = {"legacy": audit_teachers(read_jsonl(p["legacy_labels"])),
              "visited_occurrences": pool["visited_occurrences"], "unique_visited_states": pool["unique_visited_states"],
              "eligible_states": pool["eligible_unique_states"], "selected_states": len(cells),
              "selected_stage_counts": dict(Counter(c["stage"] for c in cells)),
              "competence": competence, "cells": cells,
              "confirmed_labels": len(labels), "new_exact_observation_states": sum(c["new_exact_observation"] for c in cells),
              "stage_C_signal_gate_passed": ready, "stage_C_training_authorized_by_this_script": False,
              "decision": "SIGNAL_READY_FOR_SEPARATE_C_PROTOCOL" if ready else "SIGNAL_NOT_ESTABLISHED_NO_C_TRAINING",
              "limitations": ["exact observation novelty is not new relations or new task generalization",
                              "two correlated backbone policies are not independent ground truth",
                              "old Phase2 generator contains known shortcut; validate repaired data separately",
                              "no final test read; no trained-policy method gain measured",
                              "safe abort rate is not an overprotection metric without a feasible continuation witness"]}
    write(args.study / "ANALYSIS.json", report)
    # Candidate evidence is retained even if competence fails; never exported as training labels.
    write(args.study / "CONFIRMED_LABELS.json", labels if ready else [])
    print(json.dumps({k: v for k, v in report.items() if k not in ("cells", "limitations")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "collect", "fork", "analyze"):
        q = sub.add_parser(command)
        q.add_argument("--study", type=Path, required=True)
        if command == "freeze":
            for name in ("source", "base", "collector", "reference", "legacy-labels"):
                q.add_argument("--" + name, type=Path, required=True)
            q.add_argument("--teachers", type=Path, nargs="+", required=True)
            q.add_argument("--k", type=int, default=32)
            q.add_argument("--max-states", type=int, default=4)
            q.add_argument("--smoke", action="store_true", help="plumbing only; cannot open the stage C signal gate")
        if command == "fork":
            q.add_argument("--policy", choices=("collector", "reference"), required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
