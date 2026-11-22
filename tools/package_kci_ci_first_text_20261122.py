"""Bundle 2026-11-22 minimal CI-first and LLM text-surface repair evidence."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"20261122_kci_ci_first_delivery"
ZIP=OUT/"KCI_CI_FIRST_LLM_TEXT_CPU_20261122.zip"
names=[
 "docs/KCI_CI_FIRST_CREDIT_LLM_TEXT_20261122_CN.md",
 "tools/kci_ci_first_preference_gate.py",
 "tools/kci_llm_text_surface.py",
 "tools/run_kci_ci_first_native_gate_cpu.py",
 "tools/audit_kci_ci_first_gate_cpu.py",
 "code/tests/test_kci_llm_view_ci_gate.py",
]
source=ROOT/"artifacts"/"20261122_kci_ci_first_sampled_gate_cpu_v1"
assert source.exists()
names += [str(p.relative_to(ROOT)).replace("\\","/")
          for p in sorted(source.rglob("*")) if p.is_file()]
OUT.mkdir(parents=True,exist_ok=True)
if ZIP.exists():raise FileExistsError(ZIP)
manifest={}
with zipfile.ZipFile(ZIP,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for rel in names:
        payload=(ROOT/rel).read_bytes()
        manifest[rel]={"sha256":hashlib.sha256(payload).hexdigest(),
                       "bytes":len(payload)}
        z.writestr(rel,payload)
    z.writestr("MANIFEST_SHA256.json",
               json.dumps(manifest,indent=2,ensure_ascii=False).encode("utf8"))
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    for name,record in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==record["sha256"]
receipt={
    "status":"PASS",
    "n_files":len(names)+1,
    "zip_bytes":ZIP.stat().st_size,
    "sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
    "description":"Deterministic role text projection and CI-aware CCE gate (CPU, 3 seeds)"
}
(OUT/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
print("PACKAGE_COMPLETE",str(ZIP),json.dumps(receipt),flush=True)
