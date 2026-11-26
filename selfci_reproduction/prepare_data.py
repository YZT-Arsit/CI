"""Verify the pinned CI-RL source and prepare SelfCI's official local DatasetDict."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=here / "upstream")
    parser.add_argument("--data", type=Path, default=here / "data")
    parser.add_argument("--model", required=True, help="Path to Qwen3-4B-Instruct-2507")
    parser.add_argument("--output", type=Path, default=here / "prepared")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from src.utils.preprocess import (
        prepare_dual_feedback_onpolicy_dataset_splits_from_hf,
        prepare_grpo_dataset_splits_from_hf,
    )
    from src.utils.response_parsing import is_reasoning_model

    if is_reasoning_model(args.model):
        raise ValueError("The model path must identify this backbone as instruction-tuned.")
    for source in json.loads((args.data / "sources.json").read_text(encoding="utf-8")):
        file = args.data / source["file"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {file}")
    raw = pq.read_table(args.data / "raw/train.parquet")
    feedback = pq.read_table(args.data / "feedback/train.parquet")
    if not raw.equals(feedback.select(raw.column_names)):
        raise ValueError("Original and feedback source rows differ in content or order.")
    coverage = {"allowed": 0, "disallowed": 0}
    for idx, row in enumerate(feedback.to_pylist()):
        annotation = json.loads(row["dataset_item"])["annotation"]
        for branch in coverage:
            values = json.loads(row[f"{branch}_feedbacks"])
            if not values or set(values) != set(annotation[branch]) or not all(
                isinstance(value, str) and value.strip() for value in values.values()
            ):
                raise ValueError(f"Incomplete {branch} feedback on row {idx}")
            coverage[branch] += len(values)

    split = prepare_dual_feedback_onpolicy_dataset_splits_from_hf(
        str((args.data / "feedback").resolve()), args.model
    )
    reference = prepare_grpo_dataset_splits_from_hf(
        str((args.data / "raw").resolve()), args.model
    )
    sizes = {"train": 590, "eval": 66, "test": 73}
    split_audit = {}
    for key, size in sizes.items():
        final, baseline = split[key], reference[key]
        if len(final) != size or len(baseline) != size:
            raise ValueError(f"Wrong split size for {key}")
        for column in baseline.column_names:
            if list(final[column]) != list(baseline[column]):
                raise ValueError(f"Training and baseline data disagree on {key}/{column}")
        prompts = [hashlib.sha256(p.encode()).hexdigest() for p in final["prompt"]]
        split_audit[key] = {"rows": size, "ordered_prompt_sha256": prompts}
    split.save_to_disk(str(args.output))
    audit = dict(rows=len(raw), feedback_counts=coverage, shuffle_seed=42, splits=split_audit)
    audit_path = args.output / "source_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {sizes}; feedback entries {coverage}; saved {args.output}")


if __name__ == "__main__":
    main()
