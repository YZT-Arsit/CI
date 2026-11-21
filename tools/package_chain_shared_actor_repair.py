r"""Immutable KCI shared-actor bug repair bundle: original vs 2x2 vs fixed.

The exact original V3.1 research dataset is unchanged. This package
includes source-backed reports, 2x2 ablation, rerun five-method comparisons,
additional fixed-DPR runs, 32-scenario stress, 6 saved CPU policy weights,
and independent native replay. Not original Qwen-4B training.
"""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from hashlib import sha256
import json,re
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/"artifacts"/"20261120_v31"
ZIP=B/"KCI_SHARED_CPU_ACTOR_RULE_ACTION_INTERACTION_REPAIR.zip"
if ZIP.exists():raise FileExistsError(ZIP)
report=ROOT/"docs"/"KCI_CHAIN_SHARED_ACTOR_REPAIR_DIAGNOSIS_CN.md"
reportmeta=ROOT/"docs"/"KCI_CHAIN_SHARED_ACTOR_REPAIR_DIAGNOSIS_AUDIT.json"
assert json.loads(reportmeta.read_text(encoding="utf-8"))["status"]=="SOURCE_VERIFIED_CHAIN_ACTOR_TWO_BUG_FIX"
folders={
    "original_model_failure_diagnostic":B/"CHAIN_CPU_HELDOUT_FAILURE_DIAG_01",
    "rule_output_before_action_conditioning":B/"CHAIN_CPU_FIXED_HORN_HEAD_DIAGNOSTIC_02",
    "2x2_representation_ablation":B/"CHAIN_CPU_HORN_RULE_BY_ACTION_INTERACTION_2X2_01",
    "repaired_five_methods":B/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_ACTION_INTERACTION_01",
    "repaired_DPR_backtrack":B/"CHAIN_DPR_BACKTRACKED_ACTION_INTERACTION_REPAIRED_01",
    "32_new_variant_stress":B/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01",
}
for key,file,expected in (
    ("repaired_five_methods","INDEPENDENT_NATIVE_REPLAY_AUDIT.json","INDEPENDENT_SHARED_CPU_ACTOR_NATIVE_REPLAY_PASS"),
    ("repaired_DPR_backtrack","INDEPENDENT_NATIVE_REPLAY_AUDIT.json","INDEPENDENT_BACKTRACKED_DPR_NATIVE_REPLAY_PASS"),
    ("32_new_variant_stress","INDEPENDENT_REPLAY_AUDIT.json","INDEPENDENT_SAVED_PYTORCH_ACTOR_STRESS_NATIVE_REPLAY_PASS"),
):
    d=json.loads((folders[key]/file).read_text(encoding="utf-8"))
    assert d["status"]==expected,(key,d["status"])
source=[
    "tools/chain_shared_actor_cpu.py",
    "tools/chain_shared_actor_cpu_fixed.py",
    "tools/chain_shared_actor_cpu_action_interaction.py",
    "tools/chain_shared_actor_cpu_soft_action_interaction.py",
    "tools/chain_shared_actor_cpu_action_interaction_contract.py",
    "tools/compare_kci_chain_methods_cpu.py",
    "tools/compare_kci_chain_methods_cpu_fixed.py",
    "tools/compare_kci_chain_methods_cpu_action_interaction.py",
    "tools/compare_kci_chain_methods_cpu_soft_action_interaction.py",
    "tools/compare_kci_chain_methods_cpu_action_interaction_contract.py",
    "tools/diagnose_chain_heldout_failure_20261120.py",
    "tools/diagnose_chain_repaired_scoring.py",
    "tools/ablate_chain_representation_2x2_sft.py",
    "tools/chain_dpr_backtrack_action_interaction.py",
    "tools/compare_chain_dpr_backtracked_action_interaction.py",
    "tools/audit_chain_action_interaction_fixed.py",
    "tools/audit_chain_dpr_backtracked_action_interaction.py",
    "tools/stress_chain_repaired_heldout_variants.py",
    "tools/audit_chain_repaired_stress.py",
    "tools/write_chain_shared_actor_repair_report.py",
    "tools/package_chain_shared_actor_repair.py",
    "code/kci/upstream_downstream_chain_pilot.py",
    "code/kci/upstream_downstream_chain_pilot_public_contract.py",
    "code/kci/diverse_ci_long_horizon_v31.py",
    "code/tests/test_chain_zero_preserving_horn_repair.py",
    "code/tests/test_chain_action_interaction_repair.py",
    "code/tests/test_chain_upstream_public_resource_contract.py",
]
files={"report/"+report.name:report,"report/"+reportmeta.name:reportmeta}
for name,path in folders.items():
    for f in path.rglob("*"):
        if f.is_file():
            files["evidence/"+name+"/"+f.relative_to(path).as_posix()]=f
for name in source:files["source/"+name]=ROOT/name
# Also pin old same-environment raw baseline comparison and original dataset:
files["old_unmodified_compared_methods/RESULTS.json"]=B/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_01"/"RESULTS.json"
files["old_native_reference_ceiling/SUMMARY.json"]=B/"CHAIN_NATIVE_ORACLE_BASELINES_01"/"SUMMARY.json"
assert all(p.is_file() for p in files.values())
fingerprints={name:sha256(p.read_bytes()).hexdigest()
              for name,p in sorted(files.items())}
meta={
    "study":"KCI CPU shared agent fixed Horn logic and learned action-risk interaction",
    "original_V31_dataset_and_old_original_actor_modified":False,
    "pair_2x2_model_code_regression_verified":True,
    "native_A_R_B_C_tool_synthetic_environment_unchanged":True,
    "SFT_and_CCE_success_under_repaired_8_world_eval":[8,8,8],
    "sft_32_posthoc_variant_stress":[32,32,32],
    "CCE_algorithm_independent_benefit_proven":False,
    "trained_Qwen3_4B_or_any_LLM":False,
    "test_graphs_inspected_during_debug":True,
    "tool_observations_structured_public_rules":True,
    "source_sha256":fingerprints,
}
with ZipFile(ZIP,"x",ZIP_DEFLATED,compresslevel=8) as archive:
    for name,file in sorted(files.items()):
        raw=file.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise ValueError("Potential token-like source in "+name)
        if file.stat().st_size>30*1024*1024:
            raise ValueError("Oversized individual file "+name)
        archive.writestr(name,raw)
    archive.writestr("BUNDLE_SHA256_MANIFEST.json",
        json.dumps(meta,ensure_ascii=False,indent=2,sort_keys=True).encode("utf-8"))
with ZipFile(ZIP) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist())==len(files)+1
    assert all(sha256(archive.read(k)).hexdigest()==v
               for k,v in fingerprints.items())
receipt={
    "status":"KCI_REPAIRED_ACTOR_COMPLETE_BUNDLE_SHA256_PASS",
    "zip":str(ZIP),
    "bytes":ZIP.stat().st_size,
    "file_count":len(files)+1,
    "sha256":sha256(ZIP.read_bytes()).hexdigest(),
    "saved_repaired_PyTorch_checkpoints":len(list(
      (B/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/"SMALL_CPU_CHECKPOINTS").glob("*.pt"))),
    "no_Qwen_weights":True,
}
path=B/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/"REPAIR_PACKAGE_RECEIPT.json"
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
