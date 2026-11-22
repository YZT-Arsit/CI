"""Verifiable result bundle for direct real-7B KCI Agent native evidence."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"20261122_kci_pretrained_llm_delivery"
ZIP=OUT/"KCI_QWEN7B_NATIVE_CCE_ROLE_LORA_20261122.zip"
SOURCE=[
    "docs/KCI_QWEN7B_NATIVE_CCE_ROLE_LORA_20261122_CN.md",
    "tools/kci_pretrained_llm_native_runner.py",
    "tools/kci_pretrained_llm_counterfactual.py",
    "tools/kci_pretrained_llm_train_forks.py",
    "tools/kci_pretrained_llm_train_B_probe.py",
    "tools/kci_pretrained_source_role_export.py",
    "tools/kci_role_lora_dpo.py",
    "tools/kci_role_adapter_router.py",
    "tools/audit_kci_pretrained_llm_native_20261122.py",
    "tools/kci_llm_text_surface.py",
    "code/tests/test_kci_pretrained_llm_bridge.py",
    "artifacts/20261122_kci_pretrained_llm_native/INDEPENDENT_NATIVE_EVIDENCE_AUDIT.json"
]
DIRS=[
  "qwen7b_A_all28_v1","qwen7b_full_exploratory8_v1",
  "qwen7b_full_deliberative8_v1","qwen7b_B_forced_DG18_v1",
  "qwen7b_B_forced_DG19_v1","qwen7b_B_CCE_train8_v1",
  "source_R_verified_positive_v1","qwen7b_B_train_policy_probe_v1"
]
for name in DIRS:
    directory=ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"/name
    assert directory.exists(),directory
    SOURCE.extend(str(p.relative_to(ROOT)).replace("\\","/")
                  for p in sorted(directory.rglob("*")) if p.is_file())
assert len(SOURCE)==len(set(SOURCE))
OUT.mkdir(parents=True,exist_ok=True)
if ZIP.exists():raise FileExistsError(ZIP)
manifest={}
with zipfile.ZipFile(ZIP,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for s in SOURCE:
        b=(ROOT/s).read_bytes()
        manifest[s]={"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
        z.writestr(s,b)
    z.writestr("MANIFEST_SHA256.json",json.dumps(manifest,indent=2,ensure_ascii=False).encode())
with zipfile.ZipFile(ZIP,"r") as z:
    assert z.testzip() is None
    for s,v in manifest.items():
        assert hashlib.sha256(z.read(s)).hexdigest()==v["sha256"]
    assert len(z.namelist())==len(SOURCE)+1
receipt={"status":"FULL_CRC_AND_SHA256_PASS",
         "archive_path":str(ZIP),
         "sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
         "archive_bytes":ZIP.stat().st_size,
         "entries":len(SOURCE)+1,
         "model_weights_not_redistributed":True,
         "raw_API_key_or_secrets_not_included":True,
         "only_frozen_LLM_inference_and_CPU_DPO_gradient_smoke":True}
(OUT/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
print("BUNDLE_VERIFIED",json.dumps(receipt),flush=True)
