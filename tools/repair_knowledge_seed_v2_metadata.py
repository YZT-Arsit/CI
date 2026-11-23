"""One-time repair of wrongly named anchor metadata; no label-value changes."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path(r"D:\research\longhistory\artifacts\20261119_composition_signals\knowledge_seed_v2")
manifest_file=ROOT/"MANIFEST.sha256.json"
manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
name="answer_key_EVAL_ONLY.jsonl"
before=manifest[name]
for file,expected in manifest.items():
    if file==name:continue
    assert sha256((ROOT/file).read_bytes()).hexdigest()==expected,file
data=[json.loads(s) for s in (ROOT/name).read_text(encoding="utf-8").splitlines()]
roots={x["seed_id"]:x for x in map(json.loads,(ROOT/"roots.jsonl").read_text(encoding="utf-8").splitlines())}
assert len(data)==20
assert all(x["linked_raw_anchor_unsafe"] and x["unrelated_raw_anchor_safe"]
           and x["seed_id"] in roots
           and "linked_raw_c0_unsafe" not in x for x in data)
newhash=sha256((ROOT/name).read_bytes()).hexdigest()
assert before!=newhash
manifest[name]=newhash
manifest_file.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
log={"kind":"metadata_field_rename_only",
     "old_field_names":["linked_raw_c0_unsafe","unrelated_raw_c0_safe"],
     "new_field_names":["linked_raw_anchor_unsafe","unrelated_raw_anchor_safe"],
     "label_values_unchanged":True,"all_records":20,
     "original_sha256":before,"repaired_sha256":newhash}
path=ROOT/"METADATA_FIELD_REPAIR_LOG.json"
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(log,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"metadata_repair":"PASS","records":20,"manifest_updated":True},ensure_ascii=False))
