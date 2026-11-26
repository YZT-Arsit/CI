"""Aggregate matched paper-test runs, resampling prompts as clusters."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PAPER = {"initial": [58.08, 65.75, 36.16], "selfci": [85.21, 64.11, 55.34],
         "grpo": [69.04, 63.01, 45.21], "ci_sft": [69.86, 60.27, 40.00]}
METRICS = ["integrity_hit", "utility_hit", "complete_hit"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    by_arm, results = {}, {}
    inputs = None
    for label, paper in PAPER.items():
        folder = args.directory / label
        runs = [json.loads((folder / f"seed-{s}.json").read_text(encoding="utf-8")) for s in range(42, 47)]
        identities = [[(r["id"], hashlib.sha256(r["prompt"].encode()).hexdigest()) for r in run] for run in runs]
        if inputs is None:
            inputs = identities[0]
        assert all(i == inputs and len(i) == 73 for i in identities)
        assert len(set(inputs)) == 73
        rows = [row for run in runs for row in run]
        hits = np.array([[[r[m] for m in METRICS] for r in run] for run in runs], dtype=int)
        assert np.array_equal(hits[:, :, 0] * hits[:, :, 1], hits[:, :, 2])
        by_arm[label] = hits
        means = hits.mean(axis=(0, 1)) * 100
        results[label] = dict(
            count=hits.sum(axis=(0, 1)).tolist(), denominator=len(rows), pct=means.tolist(),
            reference_pct=paper, delta_pp=(means-paper).tolist(),
            parsed=sum(bool(r["evaluation_parsed"]) for r in rows),
            capped=sum(r["finish_reason"] == "length" for r in rows),
            distinct_outputs=sum(len({run[i]["response_raw"] for run in runs}) for i in range(73)),
        )
    rng = np.random.default_rng(42)
    idx = rng.integers(0, 73, (10000, 73))
    comparisons = {}
    for a in ["selfci", "grpo", "ci_sft"]:
        per_prompt = (by_arm[a] - by_arm["initial"]).mean(axis=0) * 100
        boot = per_prompt[idx].mean(axis=1)
        comparisons[a] = dict(delta_pp=per_prompt.mean(axis=0).tolist(),
                              prompt_bootstrap_ci95_pp=np.quantile(boot, [0.025, 0.975], axis=0).tolist())
    (args.directory / "paired_ci.json").write_text(json.dumps(dict(
        metrics=METRICS, arms=results, paired_vs_initial=comparisons,
        ci_note="Percentile interval conditional on the five seeds. Resampling unit is the test prompt."
    ), indent=2))


if __name__ == "__main__":
    main()
