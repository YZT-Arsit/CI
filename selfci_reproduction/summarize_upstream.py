"""Validate the official runner's saved answers and report within-prompt repeats."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    report = {}
    for path in sorted(args.path.rglob("responses.json")):
        key = path.parent.relative_to(args.path).as_posix()
        metrics = json.loads(path.with_name("metrics.json").read_bytes())
        rows = json.loads(path.read_bytes())
        assert metrics["num_runs"] == 5 and len(rows) == 365
        runs = [[r for r in rows if r["run_index"] == i] for i in range(1, 6)]
        prompts = [r["prompt"] for r in runs[0]]
        for run in runs:
            assert [r["id"] for r in run] == list(range(73)) and [r["prompt"] for r in run] == prompts
        counts = [sum(r[k] for r in rows) for k in ["integrity_hit", "utility_hit", "complete_hit"]]
        for v, metric in zip(counts, ["Integrity", "Utility", "Complete"]):
            assert abs(v/365 - metrics[metric]) < 1e-8
        unique = [len({run[i]["response"] for run in runs}) for i in range(73)]
        report[key] = dict(counts=counts, denominator=365, pct=[x/3.65 for x in counts],
                           parsed=sum(r["evaluation_parsed"] for r in rows),
                           distinct_responses_within_prompt=sum(unique),
                           prompts_identical_across_five=unique.count(1),
                           per_run=metrics["per_run_metrics"], sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.path / "audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
