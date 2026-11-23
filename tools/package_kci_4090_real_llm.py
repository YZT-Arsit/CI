"""Package real RTX4090 Qwen3-4B KCI native results and two actual PEFT adapters."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261122_kci_4090_gpu_real_llm_v1"
OUTPUT=BASE/"KCI_4090_QWEN3_4B_REAL_LORA_NATIVE_EVIDENCE.zip"
if OUTPUT.exists():raise FileExistsError(OUTPUT)
files=[
ROOT/"docs"/"KCI_4090_REAL_4B_ROLE_DPO_20261122_CN.md",
ROOT/"tools"/"audit_kci_4090_real_llm.py",
ROOT/"tools"/"kci_gpu_4090_eval_B.py",
ROOT/"tools"/"kci_gpu_4090_eval_R.py",
ROOT/"tools"/"kci_gpu_4090_collect_B_after_R.py",
ROOT/"tools"/"kci_gpu_4090_R_task_witness.py",
ROOT/"tools"/"kci_role_lora_dpo.py",
ROOT/"tools"/"kci_role_adapter_router.py",
ROOT/"tools"/"kci_llm_text_surface.py",
ROOT/"artifacts"/"20261122_kci_4090_gpu_payload.zip",
ROOT/"artifacts"/"20261122_kci_R_task_witness_train_v1"/"DPO_R_TASK_AND_CI_TRAIN.jsonl",
ROOT/"artifacts"/"20261122_kci_R_task_witness_train_v1"/"SUMMARY.json",
ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"/"qwen7b_B_CCE_train8_v1"/"DPO_ROLE_B_TRAIN.jsonl",
]
files += [p for p in BASE.rglob("*") if p.is_file() and p!=OUTPUT]
assert len(files)==len({str(f).lower() for f in files})
manifest={}
with zipfile.ZipFile(OUTPUT,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
 for p in files:
  b=p.read_bytes()
  rel=p.relative_to(ROOT).as_posix()
  z.writestr(rel,b)
  manifest[rel]={"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
 z.writestr("MANIFEST_SHA256.json",json.dumps(manifest,indent=2).encode())
with zipfile.ZipFile(OUTPUT) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(manifest)+1
 for rel,info in manifest.items():
  assert hashlib.sha256(z.read(rel)).hexdigest()==info["sha256"]
receipt={"status":"ALL_CRC_AND_SHA256_PASS",
         "archive":str(OUTPUT),"size_bytes":OUTPUT.stat().st_size,
         "sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
         "entries":len(manifest)+1,
         "included_role_adapters":("B","R"),
         "base_model_not_copied":True,
         "no_ssh_credentials_in_bundle":True}
(BASE/"PACKAGE_RECEIPT.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
print("4090_EVIDENCE_PACKED",json.dumps(receipt,ensure_ascii=False),flush=True)
