"""Make a self-contained evidence bundle for the 2026-11-21 CPU R&D.
Existing files are read-only; never overwrite a previous package.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"20261121_kci_cpu_delivery"
ZIP=OUT/"KCI_CPU_ROLE_FACTORIZED_20261121.zip"
FILELIST=[
    "docs/KCI_CPU_ROLE_FACTORIZED_POLICY_CREDIT_20261121_CN.md",
    "code/tests/test_kci_exact_policy_continuation_cpu.py",
    "tools/kci_exact_policy_continuation_cpu.py",
    "tools/run_kci_cpu_policy_conditioned_gate.py",
    "tools/run_kci_role_local_credit_cpu.py",
    "tools/run_kci_calibrated_upstream_cpu.py",
    "tools/run_kci_downstream_first_cpu.py",
    "tools/run_kci_downstream_residual_cpu.py",
    "tools/audit_kci_cpu_direct_research_20261121.py",
]
ARTIFACTS=(
    "20261121_kci_policy_conditioned_cpu_v1",
    "20261121_kci_role_local_credit_cpu_v1",
    "20261121_kci_calibrated_upstream_cpu_v1",
    "20261121_kci_downstream_first_cpu_v1",
    "20261121_kci_downstream_residual_cpu_v1",
)
for folder in ARTIFACTS:
    p=ROOT/"artifacts"/folder
    assert p.exists(),p
    FILELIST.extend(str(x.relative_to(ROOT)).replace("\\","/")
                    for x in sorted(p.rglob("*")) if x.is_file())
assert len(FILELIST)==len(set(FILELIST))
OUT.mkdir(parents=True,exist_ok=True)
if ZIP.exists():raise FileExistsError(ZIP)
manifest={}
with zipfile.ZipFile(ZIP,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for rel in FILELIST:
        data=(ROOT/rel).read_bytes()
        manifest[rel]={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
        z.writestr(rel,data)
    z.writestr("MANIFEST_SHA256.json",
               json.dumps(manifest,ensure_ascii=False,indent=2).encode("utf8"))
with zipfile.ZipFile(ZIP,"r") as z:
    assert z.testzip() is None
    for rel,info in manifest.items():
        assert hashlib.sha256(z.read(rel)).hexdigest()==info["sha256"]
    assert len(z.namelist())==len(FILELIST)+1
(OUT/"PACKAGE_RECEIPT.json").write_text(
    json.dumps({"path":str(ZIP),"sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
                "entries":len(FILELIST)+1,"status":"ZIP_CRC_AND_SHA256_PASS"},
               indent=2,ensure_ascii=False)+"\n",encoding="utf8")
print("BUNDLE_OK",ZIP,"BYTES",ZIP.stat().st_size,"FILES",len(FILELIST)+1,
      "SHA256",hashlib.sha256(ZIP.read_bytes()).hexdigest(),flush=True)
