"""2026-11-21 KCI-A-R-B-C CPU closed-loop policy-conditioned CCE experiment.

All arms share the original small shared neural actor, train/development splits,
real native transitions and publicly observable cost contract. No LLM/GPU.
Dynamic Qpi teachers are computed from actual FROZEN current-actor continuations,
never hidden test labels. Deliberately separate legal CCE from violating DPR.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "tools")]
import compare_kci_chain_methods_cpu_action_interaction as T
from chain_shared_actor_cpu_action_interaction import (
    SharedHornActor, valid_decision_candidates, score_actions, run_episode,
)
from kci.diverse_ci_branch_env import digest
from kci_exact_policy_continuation_cpu import (
    environment_public, FrozenPolicyValue, enumerate_decisions,
    policy_preference_points,
)

torch.set_num_threads(2)
SEEDS = (7, 1234, 2025)
METHODS = (
    "action_sft", "sft_counterfactual_coverage", "qstar_dpo",
    "qpi_joint", "qpi_stagewise",
)
WARMUP = 30
FINETUNE = 120
REFRESH = 30
GAP = 0.004
OUT = ROOT / "artifacts" / "20261121_kci_policy_conditioned_cpu_v1"

def build_common_inputs():
    routine, expanded, high, counts = T.build_expert_dataset(T.GROUP["train"])
    # Same native data and labels. The only observation change is public
    # setup/cost information that later Coordinator already possesses.
    public_A = {}
    for row in T.GROUP["train"]:
        raw = T.environment(row)
        public = environment_public(row)
        s0, a0 = raw.transition(raw.initial(), "A_READ_VERIFIED_PACKAGES")
        s1, a1 = public.transition(public.initial(), "A_READ_VERIFIED_PACKAGES")
        assert a0["verdict"] == a1["verdict"] == "VALID"
        public_A[digest(raw.observation(s0))] = public.observation(s1)
    def enrich(points):
        results = []
        for p in points:
            new = deepcopy(p)
            if new["actor"] == "RecordsAgent":
                h = new["obs_sha256"]
                assert h in public_A, ("Missing training A observation", h)
                new["obs"] = deepcopy(public_A[h])
                new["obs_sha256"] = digest(new["obs"])
            results.append(new)
        return results
    return enrich(routine), enrich(expanded), enrich(high), counts


def pool_native_states():
    pool = []
    overall = Counter()
    for row in T.GROUP["train"]:
        e = environment_public(row)
        points, audit = enumerate_decisions(e)
        pool.append((row, e, points))
        overall.update(audit)
    return pool, dict(overall)


def make_policy_data(frozen_net, pool, include_repair=True):
    all_pairs = defaultdict(list)
    audits = Counter()
    per_case = []
    for row, env, points in pool:
        buckets, record = policy_preference_points(
            env, frozen_net, points, min_abs_gap=GAP,
            include_repair=include_repair)
        for stage, items in buckets.items():
            for obs, good, bad, margin, key in items:
                assert good != bad
                assert stage == "repair" or stage in ("A_SEND", "preflight", "release")
                all_pairs[stage].append(T.point(
                    obs, good, rejected=bad, weight=max(float(margin), GAP),
                    origin=("DPR_native_immediate_violation" if stage == "repair"
                            else "PCCE_exact_frozen_policy_"+stage)))
        audits.update(record)
        per_case.append({
            "world": row["world"],
            "native_observation_points": len(points),
            "CCE_legal_pairs": sum(len(buckets[x]) for x in ("A_SEND", "preflight", "release")),
            "DPR_immediate_violation_pairs": len(buckets["repair"]),
            "qpi_states_evaluated": record["exact_policy_states_evaluated"],
        })
    return dict(all_pairs), dict(audits), per_case


def make_caches(pairs, reference):
    prepared = {}
    for stage, points in pairs.items():
        if not points:
            continue
        cache = T.feature_cache(points)
        prepared[stage] = (cache, T.ref_diff(reference, cache))
    return prepared


def preference_loss(net, prepared, active):
    losses = []
    for stage in active:
        if stage not in prepared:
            continue
        cache, ref_values = prepared[stage]
        losses.append(T.supervised_loss(net, cache, "dpo", ref_values))
    if not losses:
        return None
    return torch.stack(losses).mean()


def greedy_eval(net, rows):
    results = []
    for row in rows:
        e = environment_public(row)
        episode = run_episode(e, net, greedy=True)
        a = next((x["action"] for x in episode["actions"]
                  if x["stage"] == "A_SEND" and x["learned_decision"]), "")
        choice = a.removeprefix("A_SEND_")
        q = row["upstream_A_values_given_exact_B_continuation"]
        regret = float(max(q.values()) - q[choice]) if choice in q else None
        exact = FrozenPolicyValue(e, net).value(e.initial())
        results.append({
            "world": row["world"], "source_mode": row["source_mode"],
            "success": bool(episode["success"]),
            "ci_violation": bool(episode["violation"]),
            "greedy_return": float(episode["return"]),
            "greedy_A_action": choice,
            "strict_A_value_order": abs(q["PACKAGE_A"]-q["PACKAGE_B"]) > 1e-8,
            "A_reference_native_regret": regret,
            "policy_expected_return": float(exact[0]),
            "policy_expected_safe_success": float(exact[1]),
            "policy_expected_CI_violation": float(exact[2]),
            "greedy_policy_choice_count": episode["n_policy_choices"],
            "greedy_native_actions": len(episode["actions"]),
        })
    strict = [x for x in results if x["strict_A_value_order"]]
    return {
        "n": len(results),
        "greedy_safe_success": sum(x["success"] for x in results),
        "greedy_CI_violations": sum(x["ci_violation"] for x in results),
        "mean_greedy_return": float(np.mean([x["greedy_return"] for x in results])),
        "mean_policy_expected_return": float(np.mean([x["policy_expected_return"] for x in results])),
        "mean_policy_expected_CI_violation": float(np.mean([x["policy_expected_CI_violation"] for x in results])),
        "mean_policy_expected_success": float(np.mean([x["policy_expected_safe_success"] for x in results])),
        "A_strict_optimal": sum(x["A_reference_native_regret"] < 1e-8 for x in strict),
        "A_strict_total": len(strict),
        "mean_A_reference_regret": float(np.mean([x["A_reference_native_regret"] for x in results])),
        "per_world": results,
    }


def train_one(seed, method, common, expanded, high, pool):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    net = SharedHornActor()
    optimizer = torch.optim.Adam(net.parameters(), lr=T.LR)
    cache_common = T.feature_cache(common)
    cache_extra = T.feature_cache(expanded)
    cache_qstar = T.feature_cache(high)
    for step in range(WARMUP):
        T.optimize_sft(net, cache_common, optimizer)
    reference = deepcopy(net).eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    ref_qstar = T.ref_diff(reference, cache_qstar)
    eval_before = {name: greedy_eval(net, rows)
                   for name, rows in T.GROUP.items() if name != "train"}
    cached = {}
    refresh_logs = []
    loss_log = []
    started = time.time()
    for step in range(FINETUNE):
        if method.startswith("qpi_") and step % REFRESH == 0:
            frozen = deepcopy(net).eval()
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
            pairs, audit, cases = make_policy_data(frozen, pool, include_repair=True)
            cached = make_caches(pairs, reference)
            refresh_logs.append({
                "step": step,
                "pairs": {key: len(val) for key, val in pairs.items()},
                "qpi_native_audit": audit,
                "per_case": cases,
                "teacher_policy_frozen_at_update": WARMUP + step,
            })
        optimizer.zero_grad()
        if method == "action_sft":
            loss = T.supervised_loss(net, cache_common, "sft")
        elif method == "sft_counterfactual_coverage":
            loss = T.supervised_loss(net, cache_extra, "sft")
        elif method == "qstar_dpo":
            loss = (T.supervised_loss(net, cache_qstar, "dpo", ref_qstar) +
                    0.35*T.supervised_loss(net, cache_common, "sft"))
        elif method == "qpi_joint":
            pl = preference_loss(net, cached, ("A_SEND", "preflight", "release", "repair"))
            assert pl is not None
            loss = pl + 0.35*T.supervised_loss(net, cache_common, "sft")
        elif method == "qpi_stagewise":
            # Improve executable downstream skills before updating upstream.
            # Every step still uses ONE shared actor; no separately trained agent.
            active = (("preflight", "release", "repair") if step < 60
                      else ("A_SEND", "preflight", "release", "repair"))
            pl = preference_loss(net, cached, active)
            assert pl is not None
            loss = pl + 0.35*T.supervised_loss(net, cache_common, "sft")
        else:
            raise ValueError(method)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        optimizer.step()
        if (step+1) in (1, 30, 60, 90, 120):
            loss_log.append({
                "step": step+1, "loss": float(loss.detach()),
                "preclip_gradient_norm": float(norm),
            })
    eval_after = {name: greedy_eval(net, rows)
                  for name, rows in T.GROUP.items() if name != "train"}
    model_path = OUT / "CHECKPOINTS" / (method + "_seed" + str(seed) + ".pt")
    torch.save({
        "model_state_dict": net.state_dict(), "method": method, "seed": seed,
        "warmup_updates": WARMUP, "subsequent_updates": FINETUNE,
        "public_contract_for_A": True,
        "training_rule_families": sorted({w["graph"] for w in T.GROUP["train"]}),
        "no_test_teacher_data": True,
    }, model_path)
    ckhash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    record = {
        "method": method, "seed": seed,
        "time_seconds": round(time.time()-started, 4),
        "checkpoint": model_path.name, "checkpoint_sha256": ckhash,
        "before": eval_before, "after": eval_after,
        "refresh_log": refresh_logs,
        "loss_checkpoints": loss_log,
        "weight_updates_after_common_SFT": FINETUNE,
        "teacher_type": ("recomputed_frozen_actual_policy_Qpi"
                         if method.startswith("qpi_") else
                         "native_exact_Qstar" if method=="qstar_dpo" else
                         "expert_behavior_labels"),
    }
    print("RESULT", method, seed,
          "test", eval_after["exploratory_test"]["greedy_safe_success"],
          "A", eval_after["exploratory_test"]["A_strict_optimal"],
          "expR", round(eval_after["exploratory_test"]["mean_policy_expected_return"], 5),
          "greedyR", round(eval_after["exploratory_test"]["mean_greedy_return"], 5),
          "sec", record["time_seconds"], flush=True)
    return record


def summarize(records):
    summary = {}
    for mode in METHODS:
        runs = [x for x in records if x["method"] == mode]
        summary[mode] = {}
        for split in ("dev", "exploratory_test"):
            summary[mode][split] = {}
            for key in ("greedy_safe_success", "greedy_CI_violations",
                        "A_strict_optimal", "A_strict_total",
                        "mean_A_reference_regret", "mean_policy_expected_return",
                        "mean_policy_expected_CI_violation",
                        "mean_greedy_return"):
                vals = [r["after"][split][key] for r in runs]
                summary[mode][split][key] = vals
    return summary


def main():
    if OUT.exists():
        raise FileExistsError("Never overwrite a prior research trial: "+str(OUT))
    (OUT / "CHECKPOINTS").mkdir(parents=True)
    common, expanded, high, expert_counts = build_common_inputs()
    pool, pool_counts = pool_native_states()
    print("EXPERT_COUNTS", len(common), len(expanded), len(high), expert_counts, flush=True)
    print("ALL_TRAIN_STATE_COUNTS", pool_counts, flush=True)
    results = []
    for method in METHODS:
        for seed in SEEDS:
            results.append(train_one(seed, method, common, expanded, high, pool))
    payload = {
        "status": "COMPLETED_NATIVE_CPU_STOCHASTIC_POLICY_PILOT",
        "source": "existing dynamic A-R-B-C native pilot; public A cost contract",
        "model": "4.5k trainable shared CPU Horn actor; no LLM",
        "train_worlds": 16, "dev_worlds": 4, "exploratory_test_worlds": 8,
        "test_is_not_blind": True,
        "all_source_rules_fictional_synthetic": True,
        "seeds": SEEDS, "methods": METHODS,
        "warmup_SFT_updates": WARMUP, "comparison_updates": FINETUNE,
        "qpi_refresh_period_updates": REFRESH,
        "min_pairwise_qpi_gap": GAP,
        "one_shared_actor_all_roles": True,
        "A_receives_only_already_public_cost_contract": True,
        "CCE_requires_two_immediately_VALID_actions": True,
        "DPR_requires_ACTUAL_immediate_violation_vs_valid_source_action": True,
        "no_target_oracle_in_policy_inputs": True,
        "qpi_exact_enumeration_not_4B_sampling": True,
        "not_cost_matched_Qpi_vs_Qstar_supervision": True,
        "reference_Qstar_labels_cost_search": "original expert labels reused",
        "training_state_sources": {
            "common":len(common), "counterfactual_SFT":len(expanded),
            "qstar_preferences":len(high),
            "native_reachable_state_count":pool_counts,
        },
        "summary": summarize(results),
        "per_seed": results,
    }
    p = OUT / "RESULTS.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
    print("FINAL_SUMMARY",json.dumps(payload["summary"],ensure_ascii=False),flush=True)
    print("RESULT_FILE",p,flush=True)


if __name__ == "__main__":
    main()
