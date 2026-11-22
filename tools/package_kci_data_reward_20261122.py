"""Pack the exact 2026-11-22 KCI reward/data readiness audit and its sources."""
from __future__ import annotations
from pathlib import Path
import zipfile,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"20261122_kci_reward_data_delivery"
ZIP=OUT/"KCI_LLM_DATA_REWARD_AUDIT_20261122.zip"
FILES=[
 "docs/KCI_LLM_DATA_REWARD_READINESS_20261122_CN.md",
 "tools/audit_kci_llm_data_reward_20261122.py",
 "tools/audit_kci_disclosure_balance_20261122.py",
 "code/tests/test_kci_reward_data_alignment_cpu.py",
 "artifacts/20261122_kci_data_reward_alignment_audit_cpu_v1/AUDIT.json",
 "artifacts/20261122_kci_data_reward_alignment_audit_cpu_v1/REWARD_PENALTY_SENSITIVITY.json",
 "artifacts/20261122_kci_data_reward_alignment_audit_cpu_v1/RAW_AUDIT.json",
 "artifacts/20261122_kci_safe_disclosure_balance_CPU_v1/RESULTS.json",
]
OUT.mkdir(parents=True,exist_ok=True)
if ZIP.exists():raise FileExistsError(ZIP)
manifest={}
with zipfile.ZipFile(ZIP,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for rel in FILES:
  raw=(ROOT/rel).read_bytes()
  manifest[rel]={"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
  z.writestr(rel,raw)
 z.writestr("MANIFEST_SHA256.json",json.dumps(manifest,indent=2).encode("utf8"))
with zipfile.ZipFile(ZIP) as z:
 assert z.testzip() is None
 for rel,info in manifest.items():
  assert hashlib.sha256(z.read(rel)).hexdigest()==info["sha256"]
 receipt={"status":"COMPLETE_CRC_AND_SHA256_PASS","archive_sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
          "archive_size_bytes":ZIP.stat().st_size,"entries":len(manifest)+1}
(OUT/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
print("AUDIT_PACKED",str(ZIP),json.dumps(receipt),flush=True)
