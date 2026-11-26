"""Paired differences for native CI-RL responses and KCI validation source groups."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "results"


def paired_interval(differences):
    values = np.asarray(differences, dtype=float)
    indices = np.random.default_rng(20261125).integers(len(values), size=(10000, len(values)))
    resampled = values[indices].mean(axis=1)
    return dict(n=len(values), delta_pp=(100 * values.mean(axis=0)).tolist(),
                ci95_pp=(100 * np.quantile(resampled, [0.025, 0.975], axis=0)).tolist())


def native_array(folder):
    rows = json.loads((folder / "responses.json").read_bytes())
    rows.sort(key=lambda r: (r["id"], r["run_index"]))
    assert [(r["id"], r["run_index"]) for r in rows] == [(i, j) for i in range(73) for j in range(1, 6)]
    hits = [[r[k] for k in ("integrity_hit", "utility_hit", "complete_hit")] for r in rows]
    return [r["prompt"] for r in rows], np.asarray(hits, dtype=float).reshape(73, 5, 3).mean(axis=1)


def main():
    report = {"native_ci_rl": {}, "kci": {}}
    for initial in sorted((ROOT / "upstream").glob("*_initial")):
        student = initial.with_name(initial.name.removesuffix("_initial") + "_selfci")
        if not (student / "responses.json").exists():
            continue
        prompts, baseline = native_array(initial)
        student_prompts, adapted = native_array(student)
        assert prompts == student_prompts
        report["native_ci_rl"][initial.name.removesuffix("_initial")] = paired_interval(adapted - baseline)
    for path in sorted(ROOT.glob("kci*_paired/summary.json")):
        scores = json.loads(path.read_bytes())
        groups = [v for k, v in scores["slices"].items() if k.startswith("group/")]
        assert len(groups) == 40
        diff = [g["selfci"].get("accepted", 0) / sum(g["selfci"].values())
                - g["base"].get("accepted", 0) / sum(g["base"].values()) for g in groups]
        total = scores["slices"]["all"]
        transitions = total["transitions"]
        gains = sum(n for k, n in transitions.items() if k.split("->")[0] != "accepted" and k.endswith("->accepted"))
        losses = sum(n for k, n in transitions.items() if k.startswith("accepted->") and k.split("->")[1] != "accepted")
        report["kci"][path.parent.name] = dict(group_macro=paired_interval(diff), groups_up=sum(v>0 for v in diff),
                 groups_down=sum(v<0 for v in diff), groups_tied=diff.count(0), record_gains=gains, record_losses=losses)
    (ROOT / "paired_intervals.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
