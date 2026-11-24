"""Bound Qpi on the previously hash-selected four KC8 TRAIN forks.

Acquire missing actor distributions by largest unresolved probability mass,
never by which action would make CCE win. No training or new roots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import kci_kc8_real4b_train_diagnostic as native
from kci_pisas_model_side_bridge import FrozenSharedSemantic4B, file_sha, install_isolated_precision_matched_hf
from kci_bounded_policy_value import evaluate, compare, normalized


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def write(p, value):
    with Path(p).open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def setup(source):
    native.D = source.resolve()
    native.SRC = native.D / "KC8_NATIVE_CODE_SNAPSHOT_TRAIN_ONLY.tar.gz"
    native.CONF = native.D / "FROZEN_PROTOCOL.json"
    candidates, env_type, view, digest, _ = native.load_native_snapshot()
    cases = {c.id: c for c in candidates() if c.split == "train"}
    return cases, env_type, view, digest


def sweep(source, selected, cache):
    cases, Env, view, digest = setup(source)
    cells, frontier, views = [], {}, {}
    for cell in selected:
        env = Env(cases[cell["root_id"]], cell["history"])
        initial = native.clone_prefix(env)
        if digest(initial.__dict__) != cell["A_initial_state_SHA256"]:
            raise ValueError("initial state changed")
        branches = {}
        for action in ("SEND_A", "SEND_B"):
            after, verdict = env.transition(initial, action)
            if verdict["verdict"] != "VALID":
                raise ValueError("current A is not lawful")
            result, pending = evaluate(env, after, cache, view, digest, structural_bounds=True)
            branches[action] = result
            for key, mass in result["frontier"].items():
                frontier[key] = frontier.get(key, 0.) + mass / (len(selected) * 2)
            views.update(pending)
        cells.append({**cell, "branches": branches,
                      "comparison": compare(branches["SEND_A"], branches["SEND_B"])})
    return cells, frontier, views


def prepare(args):
    if args.study.exists():
        raise FileExistsError(args.study)
    source = args.source.resolve()
    cases, Env, view, digest = setup(source)
    folder = source / "REAL4B_TRAIN_FROZEN_DIAGNOSTIC"
    full = folder / "REAL_4B_TRAIN_FULL_MODEL_NATIVE_EPISODES.jsonl"
    selection = folder / "FROZEN_FORK_SELECTION_BEFORE_BRANCH_OUTCOMES.json"
    selected = read(selection)["selected"]
    old = read(source / "FROZEN_PROTOCOL.json")
    cache = {}
    rows = [json.loads(line) for line in full.read_text(encoding="utf8").splitlines()]
    for row in rows:
        if row["root_id"] not in cases or row["checkpoint_sha256"] != old["checkpoint_SHA256"]:
            raise ValueError("mixed checkpoint or non-TRAIN trace")
        env = Env(cases[row["root_id"]], row["history"])
        s = native.clone_prefix(env)
        for event in row["real_model_action_trace"]:
            if digest(s.__dict__) != event["native_state_before_SHA256"]:
                raise ValueError("trace prefix mismatch")
            if event["event_type"] == "chance":
                options = [n for _, tag, n in env.chance_outcomes(s)
                           if tag == event["native_action_or_chance"] and digest(n.__dict__) == event["native_state_after_SHA256"]]
                if len(options) != 1:
                    raise ValueError("chance trace mismatch")
                s = options[0]
            else:
                v = view(env.actor_observation(s))
                if v["prompt_sha256"] != event["model_prompt_SHA256"]:
                    raise ValueError("model view mismatch")
                probs = normalized(event["model_candidate_probabilities"], v["available_actions"])
                key = v["prompt_sha256"]
                if key in cache and any(abs(cache[key][a] - p) > 1e-6 for a, p in probs.items()):
                    raise ValueError("same checkpoint prompt has inconsistent distributions")
                cache[key] = probs
                s, _ = env.transition(s, event["native_action_or_chance"])
            if digest(s.__dict__) != event["native_state_after_SHA256"]:
                raise ValueError("native after-state mismatch")
    teacher_file = source / "FROZEN_KC8_TRAIN_INPUTS/TRAIN_EARLY_A_ACTION_CE_IDENTICAL_STATES.jsonl"
    teachers = {r["prompt_sha256"]: r["target_action"] for r in
                (json.loads(line) for line in teacher_file.read_text(encoding="utf8").splitlines())}
    for cell in selected:
        env = Env(cases[cell["root_id"]], cell["history"])
        cell["existing_native_CI_first_CE_action"] = teachers.get(view(env.actor_observation(native.clone_prefix(env)))["prompt_sha256"])
    paths = [full, selection, teacher_file, source / "FROZEN_PROTOCOL.json",
             source / "PRECISION_RUNTIME_ISOLATED_PATCH.json", native.SRC,
             Path(__file__), Path(__file__).with_name("kci_bounded_policy_value.py"),
             Path(__file__).with_name("kci_pisas_model_side_bridge.py"),
             Path(__file__).with_name("kci_kc8_real4b_train_diagnostic.py")]
    protocol = {"question": "Does finite frozen-model Qpi have a certified direction different from native CI-first CE?",
        "source_directory": source.as_posix(),
        "scope": "white-box TRAIN finite-policy diagnostic, not CCE superiority or black-box evidence",
        "checkpoint_sha256": old["checkpoint_SHA256"], "selected": selected,
        "max_new_candidate_forwards": 600, "max_scoring_wall_seconds": 600,
        "expansion": "largest mean unresolved probability mass across all eight branches; hash ties",
        "numeric_tolerance": 1e-6, "no_model_training": True,
        "unqueried_states": "finite native min/max over all actions gives conservative component bounds, not replacement model actions",
        "existing_policy_distributions": len(cache), "existing_episodes": len(rows),
        "inputs": {str(p.resolve()): file_sha(p) for p in paths},
        "known_cost_note": "reused actor probabilities were acquired by prior 24-episode study, not free; new GPU counters incremental only"}
    args.study.mkdir(parents=True)
    write(args.study / "KNOWN_DISTRIBUTIONS.json", cache)
    protocol["known_distributions_sha256"] = file_sha(args.study / "KNOWN_DISTRIBUTIONS.json")
    write(args.study / "PROTOCOL.json", protocol)
    cells, frontier, _ = sweep(source, selected, cache)
    write(args.study / "BEFORE.json", {"cells": cells, "frontier_size": len(frontier)})
    print(json.dumps({"cells": len(cells), "known_prompts": len(cache), "frontier_size": len(frontier),
                      "before": [(c["root_id"], c["comparison"]) for c in cells]}), flush=True)


def run(args):
    p = read(args.study / "PROTOCOL.json")
    if (args.study / "START.json").exists():
        raise FileExistsError("one fixed GPU attempt only")
    for original, expected in p["inputs"].items():
        portable = original.replace("\\", "/")
        if portable.startswith(p["source_directory"] + "/"):
            actual = args.source / portable[len(p["source_directory"]) + 1:]
        elif "/tools/" in portable:
            actual = Path(__file__).parent / portable.split("/tools/", 1)[1]
        else:
            raise ValueError("unmapped frozen input")
        if file_sha(actual) != expected:
            raise ValueError("frozen input changed: " + str(actual))
    if file_sha(args.study / "KNOWN_DISTRIBUTIONS.json") != p["known_distributions_sha256"]:
        raise ValueError("imported model distributions changed")
    if file_sha(Path(args.adapter) / "adapter_model.safetensors") != p["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if any(int(x) > 512 for x in usage.splitlines()):
        raise RuntimeError("GPU occupied; no other process interrupted")
    setup(args.source)
    patch = read(args.source / "PRECISION_RUNTIME_ISOLATED_PATCH.json")
    install_isolated_precision_matched_hf(study_dir=args.source, expected_sha256=patch["audited_HF_policy_sha256"])
    model = FrozenSharedSemantic4B(args.base, args.adapter, max_prompt_tokens=8192)
    cache = read(args.study / "KNOWN_DISTRIBUTIONS.json")
    write(args.study / "START.json", {"protocol_sha256": file_sha(args.study / "PROTOCOL.json"), "checkpoint": model.adapter_sha256})
    start, n = time.monotonic(), 0
    with (args.study / "GPU_DISTRIBUTIONS.jsonl").open("x", encoding="utf8") as output:
        while True:
            cells, frontier, views = sweep(args.source, p["selected"], cache)
            if not frontier:
                stop = "FINITE_POLICY_FULLY_ENUMERATED"
                break
            if time.monotonic() - start >= p["max_scoring_wall_seconds"]:
                stop = "WALL_BUDGET_REACHED"
                break
            key = min(frontier, key=lambda k: (-frontier[k], k))
            v = views[key]
            if model.forward_calls + len(v["available_actions"]) > p["max_new_candidate_forwards"]:
                stop = "FORWARD_BUDGET_REACHED"
                break
            before = model.costs()
            cache[key] = normalized(model.distribution(v), v["available_actions"])
            row = {"prompt_sha256": key, "actor_view": v, "distribution": cache[key],
                   "frontier_mass_before_scoring": frontier[key], "cost_before": before, "cost_after": model.costs()}
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            n += 1
            if n % 10 == 0:
                print("BOUNDED_QPI", n, model.forward_calls, "remaining_mass", sum(frontier.values()), flush=True)
    cells, frontier, _ = sweep(args.source, p["selected"], cache)
    write(args.study / "AFTER.json", {"cells": cells, "frontier_size": len(frontier), "stop": stop,
          "new_actor_distributions": n, "incremental_model_costs": model.costs(),
          "wall_seconds_excluding_load": time.monotonic() - start,
          "training_labels_exported": 0, "not_method_superiority": True})
    print("BOUNDED_QPI_DONE", stop, n, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    globals()[args.command](args)
