"""Bounded file-integrity checks and a reproducible second-iteration handoff."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    return {"path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size, "sha256": digest(path)}


def hash_shards(paths, label):
    """Each hash-only worker handles <=100 files and approximately <=32MB."""
    shards = [[]]
    current_bytes = 0
    for path in sorted(set(paths)):
        size = path.stat().st_size
        if shards[-1] and (len(shards[-1]) >= 100 or current_bytes + size > 32_000_000):
            shards.append([])
            current_bytes = 0
        shards[-1].append(path.relative_to(ROOT).as_posix())
        current_bytes += size
    rows, receipts = [], []
    for index, shard in enumerate(shards):
        request = RUN / "sealing" / f"{label}_{index:03d}.request.json"
        response = request.with_suffix(".response.json")
        write(request, shard)
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--hash", str(request), str(response)],
            cwd=ROOT, capture_output=True, text=True, timeout=9.5, check=True)
        assert not completed.stdout and not completed.stderr
        seconds = time.perf_counter() - started
        rows.extend(read(response))
        receipts.append({"label": label, "shard": index, "files": len(shard),
                         "seconds": seconds, "hard_timeout_seconds": 9.5, "status": "completed"})
    return rows, receipts


def finalize():
    (RUN / "sealing").mkdir(exist_ok=True)
    snapshot = read(RUN / "input_snapshot.json")
    previous_manifest = Path(snapshot["previous_manifest"])
    assert digest(previous_manifest) == snapshot["previous_manifest_sha256"]
    expected = {row["path"].replace("\\", "/"): row["sha256"]
                for row in read(previous_manifest)["files"]}
    expected.update({row["path"].replace("\\", "/"): row["sha256"]
                     for row in snapshot["previous_paper_files"]})
    expected.update(snapshot["previous_files_sha256"])
    old_rows, receipts = hash_shards([ROOT / name for name in expected], "protected")
    mismatches = [row["path"] for row in old_rows if row["sha256"] != expected[row["path"]]]
    assert not mismatches, mismatches

    replay = read(RUN / "final_replay/summary.json")
    freeze = read(RUN / "environment/freeze_v2_sha256.json")
    for name, value in {**replay["core_dependencies_sha256"], **freeze}.items():
        assert digest(ROOT / "code" / name) == value, name
    schema = read(RUN / "training_adapter/final_schema_check.json")
    assert schema["all_passed"]
    assert schema["source_sha256"] == digest(ROOT / "code/kci/training_data.py")
    for manifest_name, key in [("manifest.json", "files"), ("clarification_manifest.json", "new_files")]:
        for name, row in read(RUN / "source_anchors" / manifest_name)[key].items():
            assert digest(RUN / "source_anchors" / name) == row["sha256"], name
    policies = list((RUN / "text_policy_v2").glob("*/result.json"))
    assert len(policies) == 40
    for path in policies:
        assert read(path)["weights_sha256"] == digest(path.with_name("weights.npz")), path
    assert read(RUN / "text_history_stress/summary.json")["frozen_weights_all_unchanged"]
    for name, count in [("text_policy_jobs_v2_receipts.jsonl", 40),
                        ("text_prefix_jobs_v1_receipts.jsonl", 40),
                        ("final_tests_jobs_receipts.jsonl", 1)]:
        jobs = [json.loads(line) for line in (RUN / name).read_text(encoding="utf-8").splitlines()]
        assert len(jobs) == count
        assert all(row["status"] == "completed" and row["source_hashes_unchanged"]
                   and row["seconds"] <= 10 for row in jobs)
    assert read(RUN / "ollama_multi/replay_v2.json")["all_passed"]
    release = read(RUN / "ollama_multi/model_release.json")
    assert release["target_model_absent_after_release"] and not release["after"]["ps"]["models"]
    summary = read(RUN / "results_summary.json")
    assert summary["tests"] == 99
    assert replay["release_raw_sources_and_gate_all_passed"]
    assert replay["accepted_unique_preferences"] == 4286
    assert summary["readiness"]["bulk_llm_training"] == "NO_GO"
    assert "Ran 99 tests" in (RUN / "job_logs/iteration2_final_regression.stderr.txt").read_text(encoding="utf-8")
    task_checks = read(RUN / "source_task_probe/summary.json")
    assert task_checks["restricted_source_task_checks"] and not task_checks["dynamic_train_ready"]
    assert task_checks["new_ci_labels"] == task_checks["new_dynamic_preferences"] == 0
    assert digest(RUN / "source_task_probe/task_probe.py") == task_checks["probe_source_sha256"]
    for instance in task_checks["instances"]:
        assert instance["source_set_matches"] and instance["source_unique_oracle_matches"]
        for name, value in instance["input_sha256"].items():
            assert digest(RUN / "source_anchors" / name) == value
        family = instance["instance"].split("/")[0]
        assert digest(RUN / "source_task_probe" / f"{family}_result.json") == instance["result_sha256"]
        assert digest(RUN / "source_task_probe" / f"{family}_enumeration.jsonl") == instance["enumeration_sha256"]
    shapes = read(RUN / "batch_shape_probe/summary.json")
    assert shapes["simulations"] == 72 and shapes["successful_workers"] == 120
    assert shapes["all_pairs_kept_once_each_simulation"] and shapes["truncated_pairs"] == 0
    assert not shapes["actual_ddp_or_gpu_measurement"] and not shapes["model_training"]
    assert digest(RUN / "batch_shape_probe/probe.py") == shapes["script_sha256"]
    assert digest(RUN / "batch_shape_probe/plan.json") == shapes["plan_sha256"]

    sources = sorted(path for folder in [ROOT / "code", ROOT / "docs"]
                     for path in folder.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    added = [path for path in sources if path.relative_to(ROOT).as_posix() not in snapshot["previous_files_sha256"]]
    patch = []
    for path in added:
        name = path.relative_to(ROOT).as_posix()
        patch.extend(difflib.unified_diff([], path.read_text(encoding="utf-8").splitlines(keepends=True),
                                         fromfile="/dev/null", tofile=f"b/{name}"))
    (RUN / "changes.patch").write_text("".join(patch), encoding="utf-8")
    write(RUN / "changed_files.json", {"existing_modified": [], "added": [record(p) for p in added]})
    for path in sources:
        target = RUN / "final_source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    write(RUN / "FINAL_VALIDATION.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "all_passed": True,
        "previous_manifest_files_verified": len(read(previous_manifest)["files"]),
        "paper_files_verified": len(snapshot["previous_paper_files"]),
        "all_distinct_protected_files_verified": len(old_rows), "protected_mismatches": [],
        "previous_manifest_sha256": digest(previous_manifest),
        "final_core_dependency_and_environment_hashes_match": True,
        "final_training_adapter_schema_hash_matches": True,
        "source_anchor_and_clarification_manifests_match": True,
        "all_40_frozen_weights_match_original_training_result": True,
        "fresh_replay_accepted_pairs": 4286, "regression_tests": 99,
        "ollama_replay_all_passed": True, "ollama_released_at_recorded_cleanup": True,
        "restricted_source_task_instances": 2, "source_task_candidates": 4000,
        "source_task_inputs_and_outputs_hashes_match": True,
        "batch_shape_simulations": 72, "batch_shape_truncated_pairs": 0,
        "batch_shape_plan_and_script_hashes_match": True,
        "old_code_and_docs_modified": [], "new_code_and_docs_files": len(added),
        "copied_source_files": len(sources), "integrity_worker_receipts": receipts,
        "scope": "file integrity and saved evidence consistency, not independent scientific validation",
        "bulk_llm_training": "NO_GO"})

    # Hash all deliverables, including validation and protected-input receipts.
    # Hash-worker outputs for this phase are excluded to avoid recursive manifests.
    paths = sources + [path for path in RUN.rglob("*") if path.is_file()
                       and "__pycache__" not in path.parts and path.name != "final_manifest.json"
                       and not (path.parent == RUN / "sealing" and path.name.startswith("seal_"))]
    rows, seal_receipts = hash_shards(paths, "seal")
    write(RUN / "final_manifest.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "path_base": str(ROOT), "files": rows, "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "hash_worker_receipts": seal_receipts,
        "exclusions": ["self", "__pycache__", "sealing/seal_*.request.json", "sealing/seal_*.request.response.json"],
        "previous_manifest_sha256": digest(previous_manifest),
        "scope": "SHA256 change detection; not cryptographic tamper-proof archival"})
    print(json.dumps({"protected": len(old_rows), "new_source_files": len(added),
                      "manifest_files": len(rows), "max_hash_worker_seconds":
                      max(row["seconds"] for row in receipts + seal_receipts), "all_passed": True}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--hash":
        write(Path(sys.argv[3]), [record(ROOT / name) for name in read(Path(sys.argv[2]))])
    else:
        finalize()
