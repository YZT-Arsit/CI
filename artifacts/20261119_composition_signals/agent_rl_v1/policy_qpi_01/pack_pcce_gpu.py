"""Build isolated PCCE GPU package; no model weights or credentials."""
from pathlib import Path
from hashlib import sha256
import json,zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
assert ROOT.name=="longhistory"
paths=list((ROOT/"code"/"kci").rglob("*.py"))
paths.extend([
    ROOT/"code"/"experiments"/"__init__.py",
    ROOT/"code"/"experiments"/"receipt_dataset.py",
    ROOT/"tools"/"run_kci_agent_rl.py",
    ROOT/"tools"/"audit_kci_agent_rl.py",
    ROOT/"tools"/"audit_kci_policy_qpi.py",
    ROOT/"tools"/"audit_kci_policy_cce.py",
    HERE/"PCCE_PLAN.md"
])
paths.extend(p for p in (ROOT/"artifacts"/"20261113_iteration2"/"source_anchors").rglob("*") if p.is_file())
indexed={p.relative_to(ROOT).as_posix():p for p in paths}
assert len(indexed)==len(paths)
target=HERE/"gpu_pcce_payload.zip"
if target.exists():raise FileExistsError(target)
with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for rel,p in sorted(indexed.items()):z.write(p,rel)
stat={"sha256":sha256(target.read_bytes()).hexdigest(),
      "bytes":target.stat().st_size,
      "n_files":len(indexed),
      "files":{rel:sha256(p.read_bytes()).hexdigest() for rel,p in sorted(indexed.items())},
      "contains_credentials":False,
      "contains_model_weights":False,
      "GPU_training_not_automatically_started":True}
(HERE/"PCCE_GPU_MANIFEST.json").write_text(json.dumps(stat,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in stat.items() if k!="files"}))
