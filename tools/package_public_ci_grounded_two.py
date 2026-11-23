r"""Bundle tested W01/R03 corrected synthetic CI task data and source, no secrets."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,re

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"artifacts"/"20261119_composition_signals"/"public_workflow_grounded_W01_R03_v1"
TARGET=ROOT/"artifacts"/"20261119_composition_signals"/"KCI_W01_R03_GROUNDED_REPAIR_V1.zip"
if TARGET.exists():raise FileExistsError(TARGET)
m=json.loads((DATA/"MANIFEST_SHA256.json").read_text(encoding="utf-8"))
for name,h in m.items():
    assert sha256((DATA/name).read_bytes()).hexdigest()==h,("dataset file hash mismatch",name)
ind=json.loads((DATA/"INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))
assert ind["status"]=="INDEPENDENT_CPU_AUDIT_PASS"
assert ind["counts"]["negative_controls_checked"]==8
assert len(ind["negative_controls"])==8 and all(x["passed"] for x in ind["negative_controls"])
source=[
    ROOT/"code"/"kci"/"knowledge_world_seed_v2.py",
    ROOT/"code"/"kci"/"public_workflow_ci.py",
    ROOT/"code"/"kci"/"public_workflow_ci_grounded.py",
    ROOT/"code"/"tests"/"test_public_workflow_ci_grounded.py",
    ROOT/"code"/"tests"/"test_public_workflow_ci.py",
    ROOT/"code"/"tests"/"test_knowledge_world_seed_v2.py",
    ROOT/"tools"/"build_public_ci_grounded_two.py",
    ROOT/"tools"/"audit_public_ci_grounded_two.py",
]
from_public={
    "THIRD_PARTY_LICENSES/AGENTDOJO_MIT_LICENSE.txt":
        Path(r"D:\research\kci_public_refs_20261119\agentdojo\LICENSE"),
    "THIRD_PARTY_LICENSES/TAU2_MIT_LICENSE.txt":
        Path(r"D:\research\kci_public_refs_20261119\tau2-bench\LICENSE"),
}
files={
    **{"dataset/"+p.name:p for p in DATA.iterdir() if p.is_file()},
    **{p.relative_to(ROOT).as_posix():p for p in source},
    **from_public,
}
assert all(p.is_file() for p in files.values())
hashes={k:sha256(p.read_bytes()).hexdigest() for k,p in sorted(files.items())}
bundle_manifest={
    "name":"KCI W01 and R03 grounded synthetic task repair v1",
    "research_status":"DEVELOPMENT_ONLY; NOT PAPER-GRADE; NOT TRAINING_READY",
    "roots":2,"scenarios":16,"contrasts":6,"traces":12,
    "full_native_replay":"PASS","negative_controls":"8/8 PASS",
    "source_sha256":hashes,
}
with ZipFile(TARGET,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,p in sorted(files.items()):
        data=p.read_bytes()
        # Only bundled source and synthetic research data; do not include any
        # credential or other user's files. Avoid a false positive on 'task-'.
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",data):
            raise RuntimeError("Secret-like byte pattern in "+name)
        z.writestr(name,data)
    z.writestr("BUNDLE_SHA256_MANIFEST.json",
               json.dumps(bundle_manifest,ensure_ascii=False,indent=2,sort_keys=True).encode())
with ZipFile(TARGET) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    for name,h in hashes.items():
        assert sha256(z.read(name)).hexdigest()==h,name
receipt={
    "status":"ZIP_VERIFIED","file":str(TARGET),
    "bytes":TARGET.stat().st_size,"SHA256":sha256(TARGET.read_bytes()).hexdigest(),
    "entries":len(files)+1,
    "n_roots":2,"n_worlds":16,"n_pairs":6,"n_witnesses":12,
    "all_8_negative_controls_pass":True,"API_secrets_included":False,
    "public_upstream_harness_not_executed":True,
    "human_norm_certification":False,"model_training_done":False
}
file=DATA/"PACKAGE_RECEIPT.json"
if file.exists():raise FileExistsError(file)
file.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
