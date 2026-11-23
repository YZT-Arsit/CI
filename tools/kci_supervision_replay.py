"""Run the v0.7 supervision pipeline against immutable historical TRAIN traces.

No model loading, no resampling and no DEV/locked inputs. Missing reference
files stay missing; they are not synthesized from the collector.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from kci_cce_discovery import ROOT, digest, file_sha, read_jsonl, replay, compare_branches, audit_teachers
from kci_unique_supervision import summarize


def local_input(remote, expected):
    suffix = remote.replace("\\", "/").split("/artifacts/", 1)[1]
    path = ROOT / "artifacts" / suffix
    if file_sha(path) != expected:
        raise ValueError("historical input hash changed: " + str(path))
    return path


def replay_row(state, row):
    env, native = replay(state["context"], state["prefix"])
    if not row["events"] or row["events"][0]["action"] != row["action"]:
        raise ValueError("missing or mismatched forced action")
    for i, event in enumerate(row["events"]):
        if digest(asdict(native)) != event["before_hash"] or digest(env.observation(native)) != event["observation_hash"]:
            raise ValueError("branch observation/state mismatch")
        native, verdict = env.transition(native, event["action"])
        if digest(asdict(native)) != event["after_hash"] or verdict != event["verdict"]:
            raise ValueError("native transition mismatch")
        if i == 0 and verdict["verdict"] != "VALID":
            raise ValueError("upstream fork not currently legal")
    if (digest(asdict(native)) != row["terminal_state_hash"]
            or native.violation != row["ci_violation"]
            or bool(native.done and native.success and not native.violation) != row["ci_safe_success"]):
        raise ValueError("terminal mismatch")


def run(study, output):
    if output.exists():
        raise FileExistsError(output)
    inputs = {}
    def read(path):
        inputs[str(path.resolve())] = file_sha(path)
        return json.loads(path.read_text(encoding="utf8"))
    p = read(study / "PROTOCOL.json")
    pool = read(study / "STATE_POOL.json")
    if pool["protocol_sha256"] != file_sha(study / "PROTOCOL.json") or pool["source_episode_sha256"] != file_sha(study / "VISITED_EPISODES.jsonl"):
        raise ValueError("pool provenance mismatch")
    inputs[str((study / "VISITED_EPISODES.jsonl").resolve())] = file_sha(study / "VISITED_EPISODES.jsonl")
    teacher = {}
    for remote in p["teachers"]:
        path = local_input(remote, p["input_hashes"][remote])
        inputs[str(path)] = file_sha(path)
        for r in read_jsonl(path):
            if r["split"] != "train":
                raise ValueError("TRAIN only")
            obs = r.get("native_role_observation", r.get("role_observation"))
            action = r.get("chosen_action", r.get("teacher_actions", {}).get("action_ce_witness"))
            teacher.setdefault(digest(obs), set()).add(action)
    cells = [{**s, "existing_teacher_actions": sorted(teacher.get(s["observation_hash"], set())), "estimates": {}}
             for s in pool["states"]]
    costs, competence, replayed = {}, {}, 0
    if (study / "COLLECTION_COST.json").exists():
        costs["collection"] = read(study / "COLLECTION_COST.json")
    for name in ("collector", "reference"):
        path = study / (name + "_FORKS.jsonl")
        if not path.exists():
            continue
        marker = read(study / (name + "_FORK_START.json"))
        if marker["pool_sha256"] != file_sha(study / "STATE_POOL.json"):
            raise ValueError("fork pool provenance mismatch")
        inputs[str(path.resolve())] = file_sha(path)
        rows = read_jsonl(path)
        expected_n = len(cells) * 2 * sum(len(p[b + "_seeds"]) for b in ("discovery", "validation"))
        if len(rows) != expected_n or any(r["policy_id"] != p[name + "_sha256"] for r in rows):
            raise ValueError("incomplete/mixed checkpoint fork data")
        cost = read(study / (name + "_COST.json"))
        costs[name] = {**cost, "native_transitions": cost["branch_native_transitions"] + cost["competence_native_transitions"]}
        for cell in cells:
            if cell["context"]["spec"]["split"] != "train":
                raise ValueError("TRAIN only")
            cell["estimates"][name] = {}
            for block in ("discovery", "validation"):
                pair = [[r for r in rows if r["decision_id"] == cell["decision_id"]
                         and r["block"] == block and r["action"] == a] for a in cell["fork_actions"]]
                if any([r["seed"] for r in branch] != p[block + "_seeds"] for branch in pair):
                    raise ValueError("paired budget/seed mismatch")
                for row in pair[0] + pair[1]:
                    replay_row(cell, row)
                    replayed += 1
                cell["estimates"][name][block] = compare_branches(*pair,
                    alpha=p["alpha"] / (p["max_states"] * 2 * 2 * 5), risk_ceiling=p["risk_ceiling"])
    result = summarize(cells, competence=competence, costs=costs, smoke=p["plumbing_smoke_only"])
    legacy_path = local_input(p["legacy_labels"], p["input_hashes"][p["legacy_labels"]])
    legacy = read_jsonl(legacy_path)
    legacy_table = [{"root": r["root"], "state": r["cell_id"], **r["teacher_actions"],
                     "unique_direction_beyond_both_teachers": all(r["teacher_actions"]["CCE_confirmed"] != r["teacher_actions"][k]
                         for k in ("action_ce_witness", "terminal_K8"))} for r in legacy]
    report = {"historical_protocol_sha256": file_sha(study / "PROTOCOL.json"),
              "current_analyzer_sha256": file_sha(Path(__file__)),
              "current_gate_sha256": file_sha(ROOT / "tools/kci_unique_supervision.py"),
              "reanalysis_not_replacement_of_frozen_inference": True,
              "native_forks_replayed": replayed, "inputs": inputs,
              "legacy": {**audit_teachers(legacy), "cells": legacy_table},
              "supervision": result}
    output.mkdir(parents=True)
    (output / "RESULT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    (output / "CCE_TRAINING_LABELS.json").write_text("[]\n", encoding="utf8")
    print(json.dumps({"replayed": replayed, "same_legacy_labels": report["legacy"]["same_labels"],
                      "decision": result["decision"], "blockers": result["blockers"]}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)
