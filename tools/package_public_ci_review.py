"""Package KCI first-pass audit plus supporting evidence, leave original data untouched."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261119_composition_signals"
SRC=BASE/"public_workflow_ci_10_final"
REV=BASE/"public_workflow_ci_10_review_01"
OUT=BASE/"KCI_PUBLIC_CI_10_REVIEW_FIRST_PASS.zip"
if OUT.exists():raise FileExistsError(OUT)
review=[
    "FIRST_PASS_REVIEW_CN.md","FIRST_PASS_REVIEW_MATRIX.csv",
    "FIRST_PASS_REVIEW_MATRIX.json","DECISIONS_FOR_USER.md","REVIEW_MANIFEST.json"]
evidence=[
    "FIRST_PASS_MECHANISM_AUDIT.json",
    "FIRST_PASS_END_TO_END_BUSINESS_INTERVENTION.json",
    "FIRST_PASS_UPSTREAM_EXCERPTS.json",
    "FIRST_PASS_TEMPORAL_LANGUAGE_AUDIT.json",
    "API_LANGUAGE_DIAGNOSTIC.json",
    "API_LANGUAGE_REVIEW_REQUIRED.jsonl","HUMAN_REVIEW_PENDING.csv",
    "UPSTREAM_SOURCES.json","VALIDATION.json"]
inputs={
    **{"review/"+name:REV/name for name in review},
    **{"evidence/"+name:SRC/name for name in evidence}
}
assert all(p.is_file() for p in inputs.values())
digest={name:sha256(p.read_bytes()).hexdigest() for name,p in inputs.items()}
with ZipFile(OUT,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,p in sorted(inputs.items()):
        raw=p.read_bytes()
        assert b"sk-proj-" not in raw and b"sk-svcacct-" not in raw
        z.writestr(name,raw)
    z.writestr("MANIFEST_SHA256.json",json.dumps(
        {"files":digest,"manual_norm_approval":False,
         "source":str(SRC),"original_dataset_modified":False},
         ensure_ascii=False,sort_keys=True,indent=2).encode())
with ZipFile(OUT) as z:
    assert z.testzip() is None
    assert all(sha256(z.read(name)).hexdigest()==d for name,d in digest.items())
print(json.dumps({"file":str(OUT),"size_bytes":OUT.stat().st_size,
                  "sha256":sha256(OUT.read_bytes()).hexdigest(),
                  "entries":len(inputs)+1,"status":"VERIFIED"},ensure_ascii=False,indent=2))
