"""Export the five validation shards of KCI-Align unified roles v1 for frozen evaluation."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def save_gzip(path, rows):
    # No filename/timestamp in the header, so repeated exports have identical bytes.
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
        for row in rows:
            out.write(json_bytes(row) + b"\n")


def main():
    project = Path(__file__).resolve().parent
    workspace = project.parent
    release = workspace / "artifacts/20261125_unified_roles_v1"
    sys.path.insert(0, str(workspace / "tools/unified_data"))
    from train import execution_role, prompt

    manifest = json.loads((release / "MANIFEST.json").read_bytes())
    split = json.loads((release / "SPLITS.json").read_bytes())
    index = json.loads((release / "VALIDATION_INDEX.json").read_bytes())
    expected = set(index)
    assert len(expected) == 40 and expected == set(split["partitions"]["validation"])
    assert not expected.intersection(split["partitions"]["train"])
    registry = {root: group for group, roots in split["components"].items() for root in roots}
    blocked = {registry.get(entry["source"], entry["source"]) for entry in split["historical_and_sealed_registry"]
               if entry["status"] in ("prospective", "locked_test")}
    assert not expected.intersection(blocked)
    prompt_rows, labels, sources = {}, [], {}
    groups_by_domain = {}
    fields = ("record_id", "route", "role", "group", "domain", "evidence", "tiers",
              "value_tiers", "executable_actions", "accepted", "rejected")
    for route in ("dpo", "sft", "static_evidence", "task_dpo", "task_sft"):
        shard = "validation_" + route
        path = release / (shard + ".jsonl.gz")
        spec = manifest["shards"][shard]
        assert digest(path.read_bytes()) == spec["sha256"]
        sources[shard] = spec
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as infile:
            for line in infile:
                row = json.loads(line)
                group = row["group"]
                assert row["partition"] == "validation" and row["route"] == route
                assert group in expected and registry[row["source"]] == group
                visible = {key: row[key] for key in ("actor_context", "executable_actions")}
                tag = digest(json_bytes(visible))
                assert tag == row["tag"]
                rendered = prompt(dict(visible, route=route))
                assert rendered == prompt(row)
                item = {"prompt_id": tag, "prompt": rendered}
                assert prompt_rows.setdefault(tag, item) == item
                labels.append(dict({k: row[k] for k in fields}, prompt_id=tag,
                                   execution_role=execution_role(row)))
                groups_by_domain.setdefault(row["domain"], set()).add(group)
                count += 1
        assert count == spec["rows"]
    assert len(labels) == len({r["record_id"] for r in labels}) == 4133
    assert {r["group"] for r in labels} == expected
    out = project / "kci_inputs"
    out.mkdir(exist_ok=True)
    save_gzip(out / "prompts.jsonl.gz", prompt_rows.values())
    save_gzip(out / "labels.jsonl.gz", labels)
    metadata = dict(
        release=release.name, release_manifest_sha256=digest((release / "MANIFEST.json").read_bytes()),
        split_sha256=digest((release / "SPLITS.json").read_bytes()),
        role_prompt_code_sha256=digest((workspace / "tools/unified_data/train.py").read_bytes()),
        source_shards=sources, partition="validation", groups=40, records=len(labels),
        unique_prompts=len(prompt_rows), domains={k:len(v) for k,v in groups_by_domain.items()},
        rows_by_route=dict(Counter(r["route"] for r in labels)),
        packet_sha256={p.name:digest(p.read_bytes()) for p in (out / "prompts.jsonl.gz",out / "labels.jsonl.gz")},
    )
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
