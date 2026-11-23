"""One predeclared shared-executor sampling repair; labels and CE stay fixed."""
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from random import Random
import argparse
import json

from kci_cce_discovery import ROOT, digest, file_sha, read_jsonl, replay
from kci.diverse_ci_branch_env import digest as native_digest

DATA = Path("artifacts/20261124_cce_downstream_sft_v1")
SOURCE = Path("artifacts/20261123_cce_phase2_mechanism_v1/FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl")


def stage_cycle():
    return ("pre", "pre", "preflight", "preflight", "evidence", "evidence",
            "release", "release", "release", "attest", "post", "post", "broker", "A_READ")


def coverage_schedule(rows, *, seed, steps, batch):
    if any(r["split"] != "train" for r in rows):
        raise ValueError("TRAIN-only supervision required")
    if set(r["stage"] for r in rows) != set(stage_cycle()):
        raise ValueError("unexpected or missing stages; A_SEND remains excluded")
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        # Only the two audited scarce-decision stages gain an action stratum.
        group = row["chosen_action"] if row["stage"] in ("release", "preflight") else "all"
        buckets[row["stage"]][group][row["root"]].append(row)
    rng = Random(seed)
    for stage in sorted(buckets):
        for action in sorted(buckets[stage]):
            for root in sorted(buckets[stage][action]):
                pool = buckets[stage][action][root]
                pool.sort(key=lambda r: r["same_observation_sha256"])
                rng.shuffle(pool)
    counts, schedule = Counter(), []
    for i in range(steps * batch):
        stage = stage_cycle()[i % len(stage_cycle())]
        actions = sorted(buckets[stage])
        action = actions[counts[stage] % len(actions)]
        counts[stage] += 1
        roots = sorted(buckets[stage][action])
        root = roots[counts[(stage, action)] % len(roots)]
        counts[(stage, action)] += 1
        pool = buckets[stage][action][root]
        schedule.append(pool[counts[(stage, action, root)] % len(pool)])
        counts[(stage, action, root)] += 1
    return schedule


def write(path, data):
    with path.open("x", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def freeze(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    train_path = DATA / "SHARED_ROLE_ACTION_SFT_TRAIN_ONLY.jsonl"
    witness_path = DATA / "NATIVE_SAFE_FULL_TEAM_WITNESSES_TRAIN_ONLY.jsonl"
    old_path = DATA / "shared_native_sft_seed7/TRAINING.json"
    rows = read_jsonl(ROOT / train_path)
    old = json.loads((ROOT / old_path).read_text(encoding="utf8"))
    by_observation = {r["same_observation_sha256"]: r for r in rows}
    if len(by_observation) != len(rows):
        raise ValueError("duplicate teacher observations")
    if any(r["chosen_action"] not in r["all_executable_actions"] or
           "ABORT" not in r["all_executable_actions"] for r in rows):
        raise ValueError("invalid teacher action menu")
    specs = {r["root"]: r["native_root_spec"] for r in read_jsonl(ROOT / SOURCE)}
    candidates, seen = {}, set()
    witnesses = read_jsonl(ROOT / witness_path)
    # No model scores or DEV labels are consulted when constructing this panel.
    for w in witnesses:
        context = {"spec": specs[w["root"]], "variant": w["variant"],
                   "authorized": w["authorized"], "reorder": False, "package_swap": None}
        env, state = replay(context, [])
        prefix = []
        for event in w["events"]:
            obs = env.observation(state)
            if obs != event["role_authorized_observation"] or native_digest(asdict(state)) != event["native_state_before_sha256"]:
                raise ValueError("witness observation/state mismatch")
            label = by_observation.get(native_digest(obs))
            if label and label["chosen_action"] == event["action"]:
                seen.add(label["same_observation_sha256"])
                if state.stage != "A_READ":
                    key = (w["root"], state.stage, label["chosen_action"])
                    record = {"case_id": digest([context, asdict(state)]), "root": w["root"],
                              "stage": state.stage, "context": context, "prefix": list(prefix),
                              "observation": obs, "teacher_action": label["chosen_action"],
                              "source_teacher_observation": label["same_observation_sha256"],
                              "has_native_safe_success_witness": True}
                    if key not in candidates or record["case_id"] < candidates[key]["case_id"]:
                        candidates[key] = record
            state, verdict = env.transition(state, event["action"])
            if verdict["verdict"] != "VALID" or native_digest(asdict(state)) != event["native_state_after_sha256"]:
                raise ValueError("witness transition mismatch")
            prefix.append(event["action"])
        if not (state.done and state.success and not state.violation) or native_digest(asdict(state)) != w["native_final_hash"]:
            raise ValueError("not a genuine successful native witness")
    if seen != set(by_observation):
        raise ValueError("teacher labels missing complete replayed witnesses")
    schedule = coverage_schedule(rows, seed=7, steps=160, batch=2)
    old_counts = Counter(item["label"] for step in old["curve"] for item in step["samples"])
    panel = [candidates[k] for k in sorted(candidates)]
    scheduled = {r["same_observation_sha256"] for r in schedule}
    for case in panel:
        case["observation_used_in_candidate_schedule"] = case["source_teacher_observation"] in scheduled
    expected_stages = Counter(stage_cycle()[i % 14] for i in range(320))
    if Counter(r["stage"] for r in schedule) != expected_stages:
        raise ValueError("stage budget changed")
    paths = [SOURCE, train_path, witness_path, old_path,
             DATA / "shared_native_sft_seed7/adapter/adapter_model.safetensors",
             Path("tools/kci_executor_coverage.py"), Path("tools/kci_executor_coverage_run.py"),
             Path("tools/kci_cce_discovery.py"), Path("tools/kci_cce_full_flow_sft_train.py"),
             Path("tools/kci_cce_phase2_frozen4b_rollout.py"),
             Path("code/kci/v4_phase2_workflows.py"), Path("code/kci/agent_rl/hf_policy.py")]
    p = {"status": "FROZEN_SINGLE_EXECUTOR_COVERAGE_REPAIR", "teacher_labels_unchanged": True,
         "objective_unchanged": True, "objective": "single-label semantic-action CE, same native witness targets",
         "change": "action strata inside release/preflight; same stage quotas and total 320 observations",
         "candidate_models_allowed": 1, "steps": 160, "batch": 2, "seed": 7, "lr": 2e-5,
         "lora_rank": 8, "lora_alpha": 16, "base": old["base"],
         "initialization": "fresh LoRA, original base/seed/recipe; historical initial adapter was not saved, so bitwise identity is not established",
         "train_file": train_path.as_posix(), "baseline_adapter": (DATA / "shared_native_sft_seed7/adapter").as_posix(),
         "source_unique_observations": len(rows), "independent_train_roots": len({r["root"] for r in rows}),
         "native_success_witnesses_replayed": len(witnesses), "old_schedule_counts": dict(old_counts),
         "new_schedule_counts": dict(Counter(r["chosen_action"] for r in schedule)),
         "stage_quotas": dict(expected_stages), "unique_observations_sampled": len(scheduled),
         "schedule": [r["same_observation_sha256"] for r in schedule],
         "calibration_cases": len(panel), "calibration_model_seed": 840001,
         "calibration_panel_digest": digest(panel),
         "calibration_scope": "TRAIN native-feasible prefixes; diagnostics, not independent task generalization",
         "initial_task_contexts": [{"spec": specs[root], "variant": "risky", "authorized": auth,
                                    "reorder": False, "package_swap": None}
                                   for root in sorted(specs) for auth in (False, True)],
         "capability_gate": {"native_feasible_suffix_success_min": .90,
                             "per_stage_success_min": .80,
                             "scarce_capability_completion_min": 1.0,
                             "suffix_success_must_not_decrease": True,
                             "candidate_violations_must_not_exceed_baseline": True,
                             "legal_RAW_retention_min": .90,
                             "initial_task_success_must_not_decrease": True,
                             "UNKNOWN_max": 0},
         "no_CCE_pairs_until_capability_gate": True, "no_DPR_or_new_reward": True,
         "no_DEV_or_FINAL_for_tuning": True,
         "data_scope": "No independent roots added; rare labels remain scarce and may be memorized",
         "input_hashes": {path.as_posix(): file_sha(ROOT / path) for path in paths}}
    target.mkdir(parents=True)
    write(target / "PROTOCOL.json", p)
    write(target / "CALIBRATION_PANEL.json", panel)
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("schedule", "input_hashes", "initial_task_contexts")}, ensure_ascii=False))
