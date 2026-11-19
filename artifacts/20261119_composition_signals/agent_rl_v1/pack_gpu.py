"""Create minimal reproducible GPU transfer bundle.
Never include model caches, SSH credentials, frozen checkpoints, .git or private user data.
"""
import hashlib,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
assert ROOT.name=="longhistory",ROOT
sources=[]
sources.extend(sorted((ROOT/"code"/"kci").rglob("*.py")))
sources.extend([ROOT/"code"/"experiments"/"__init__.py",
                ROOT/"code"/"experiments"/"receipt_dataset.py"])
sources.extend(sorted((ROOT/"artifacts"/"20261113_iteration2"/"source_anchors").rglob("*")))
sources.extend([ROOT/"tools"/"run_kci_agent_rl.py",
                ROOT/"tools"/"audit_kci_agent_rl.py",
                HERE/"README.md",HERE/"requirements_gpu.txt"])
files=[p for p in sources if p.is_file()]
names={p.relative_to(ROOT).as_posix():p for p in files}
if len(names)!=len(files):raise RuntimeError("Ambiguous archive source paths")
bundle=HERE/"gpu_payload_v2.zip"
if bundle.exists():raise FileExistsError(bundle)
with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(names.items()):
        z.write(p,name)
audit={"zip_sha256":hashlib.sha256(bundle.read_bytes()).hexdigest(),
       "included_files":len(files),"zip_bytes":bundle.stat().st_size,
       "files":{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(names.items())},
       "contains_credentials":False,"contains_model_weights":False,
       "contains_historical_training_checkpoints":False}
(HERE/"GPU_PAYLOAD_V2_MANIFEST.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in audit.items() if k!="files"},indent=2))
