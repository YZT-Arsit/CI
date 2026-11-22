"""Portable result bundle for KCI 2026-11-21 sample-efficient CPU R&D.

Bundle includes code, train input native dataset, original three SFT-CF actors,
three repaired downstream source checkpoints, saved learned heads, full audit
results and a per-file checksum manifest. No overwrite.
"""
from __future__ import annotations
from pathlib import Path
import json,hashlib,zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"20261121_kci_scalable_cpu_delivery"
ZIP=OUT/"KCI_SCALABLE_CCE_CPU_20261121.zip"
SOURCES=[
    "docs/KCI_SCALABLE_SAMPLED_CCE_CPU_20261121_CN.md",
    "tools/kci_exact_policy_continuation_cpu.py",
    "tools/run_kci_budgeted_counterfactual_cpu.py",
    "tools/run_kci_budgeted_mc_dpo_train_cpu.py",
    "tools/run_kci_adaptive_mc_cee_cpu.py",
    "tools/audit_kci_sampled_budget_dpo_cpu.py",
    "tools/run_kci_native_repair_mc_bridge_cpu.py",
    "tools/audit_kci_native_repair_mc_bridge_cpu.py",
    "tools/run_kci_cpu_policy_conditioned_gate.py",
    "tools/run_kci_role_local_credit_cpu.py",
    "tools/run_kci_downstream_residual_cpu.py",
    "tools/run_kci_calibrated_upstream_cpu.py",
    "code/tests/test_kci_budgeted_counterfactual_cpu.py",
    "code/tests/test_kci_exact_policy_continuation_cpu.py",
]
OUTPUTS=[
    "artifacts/20261121_kci_budgeted_counterfactual_cpu_v2",
    "artifacts/20261121_kci_budgeted_training_cpu_v1",
    "artifacts/20261121_kci_adaptive_sampled_cee_cpu_v1",
    "artifacts/20261121_kci_budgeted_dpo_saved_audit_v1",
    "artifacts/20261121_kci_native_repair_mc_bridge_cpu_v1",
    "artifacts/20261120_v31/DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01",
]
for base in OUTPUTS:
    d=ROOT/base
    assert d.is_dir(),d
    SOURCES.extend(str(p.relative_to(ROOT)).replace("\\","/")
                   for p in sorted(d.rglob("*")) if p.is_file())
for seed in (7,1234,2025):
    SOURCES.append(
        "artifacts/20261121_kci_policy_conditioned_cpu_v1/CHECKPOINTS/sft_counterfactual_coverage_seed"+str(seed)+".pt")
    SOURCES.append(
        "artifacts/20261121_kci_downstream_residual_cpu_v1/CHECKPOINTS/A_Qpi_after_downstream_seed"+str(seed)+".pt")
assert len(set(SOURCES))==len(SOURCES)
OUT.mkdir(parents=True,exist_ok=True)
if ZIP.exists():raise FileExistsError(ZIP)
manifest={}
with zipfile.ZipFile(ZIP,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for s in SOURCES:
        content=(ROOT/s).read_bytes()
        manifest[s]={"sha256":hashlib.sha256(content).hexdigest(),"length":len(content)}
        z.writestr(s,content)
    z.writestr("MANIFEST_SHA256.json",
               json.dumps(manifest,indent=2,ensure_ascii=False).encode("utf8"))
with zipfile.ZipFile(ZIP,"r") as z:
    assert z.testzip() is None
    for s,info in manifest.items():
        assert hashlib.sha256(z.read(s)).hexdigest()==info["sha256"]
    assert len(z.namelist())==len(SOURCES)+1
receipt={"status":"CRC_SHA256_ALL_ENTRIES_PASS",
         "archive_path":str(ZIP),"archive_sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
         "archive_bytes":ZIP.stat().st_size,
         "entries":len(SOURCES)+1,
         "all_base_and_repaired_source_model_weights_present":True,
         "native_data_and_train_labels_included":True,
         "source_model_is_CPU_neural_Horn_actor_not_pretrained_4B":True}
(OUT/"RECEIPT.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
print("PACKAGE_READY",json.dumps(receipt),flush=True)
