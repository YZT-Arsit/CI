"""Bundle only KCI Qpi source, pinned native task assets, and frozen selection.

No user credentials, large checkpoint files or prior training data in package.
"""
from pathlib import Path
from hashlib import sha256
import json,zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
assert ROOT.name=="longhistory"
files=list((ROOT/"code"/"kci").rglob("*.py"))
files += [
    ROOT/"code"/"experiments"/"__init__.py",
    ROOT/"code"/"experiments"/"receipt_dataset.py",
    ROOT/"tools"/"run_kci_policy_qpi.py",
    ROOT/"tools"/"audit_kci_policy_qpi.py",
    HERE/"selection_v2"/"SELECTION.json",
    HERE/"selection_v2"/"FORKS.jsonl",
    HERE/"selection_v2"/"ELIGIBLE.jsonl",
]
files += [x for x in (ROOT/"artifacts"/"20261113_iteration2"/"source_anchors").rglob("*") if x.is_file()]
idx={p.relative_to(ROOT).as_posix():p for p in files}
assert len(idx)==len(files)
target=HERE/"gpu_qpi_payload.zip"
if target.exists():raise FileExistsError(target)
with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for rel,p in sorted(idx.items()):z.write(p,rel)
record={"SHA256":sha256(target.read_bytes()).hexdigest(),
        "bytes":target.stat().st_size,"files":len(idx),
        "contents":{rel:sha256(p.read_bytes()).hexdigest() for rel,p in sorted(idx.items())},
        "contains_user_credentials":False,
        "contains_model_weights":False,
        "input_is_known_training_prefixes_not_blind_TEST":True,
        "GPU_training_authorized":False}
(HERE/"GPU_PACKAGE_MANIFEST.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in record.items() if k!="contents"}))
