r"""Package auditable KCI Qstar/Qpi teacher mismatch and four-way CPU pilot.

Everything uses ORIGINAL original 28 source worlds, V3.1 executor, six pinned
frozen checkpoints and 12 actually trained small CPU policies. Includes
the full real native evidence, independent DFS replay, tests, scripts, and
source SHA256. No newly synthesized tasks, no GPU, no Qwen weights.
"""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from hashlib import sha256
import json,re
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/"artifacts"/"20261121_policy_gap_pilot_01"
OUT=B/"KCI_EXACT_QPI_TEACHER_MISMATCH_FOURWAY_CPU_PILOT.zip"
if OUT.exists():raise FileExistsError(OUT)
F={
 "frozen_6_checkpoints_Qpi":B/"SIX_SAVED_CHECKPOINTS_QSTAR_VS_EXACT_QPI",
 "train_only_reference_Qpi_teacher":B/"TRAIN_ONLY_QPI_CREDIT_FROM_SFT_REFERENCE",
 "four_methods_3seed_actual_trained_models":B/"FOUR_WAY_CPU_TRAINED_POLICY_RESULTS_03",
 "current_policy_exact_Qpi_pre_vs_post":B/"EXACT_QPI_BEFORE_AFTER_FOUR_METHODS_02",
 "independent_native_DFS_replay":B/"INDEPENDENT_EXHAUSTIVE_QPI_NATIVE_REPLAY",
 "DG19_B_and_source_causal_interventions":B/"DG19_PRIVILEGED_CAUSAL_ROLE_INTERVENTION",
 "DG19_source_action_definition_controls":B/"DG19_SOURCE_ONLY_INTERVENTION_DEFINITION_CHECK",
 "source_backed_research_report":B/"RESEARCH_REPORT",
}
assert json.loads((F["independent_native_DFS_replay"]/"SUMMARY.json").read_text(
    encoding="utf-8"))["status"]=="INDEPENDENT_FROZEN_AND_TRAINED_Qpi_NATIVE_DFS_REPLAY_PASS"
assert json.loads((F["source_backed_research_report"]/"RESULTS.json").read_text(
    encoding="utf-8"))["status"]=="SOURCE_AUDITED_FROZEN_QSTAR_QPI_MISMATCH_FOURWAY_CPU_COMPARISON"
scripts=(
 "tools/kci_exact_frozen_qpi_cpu.py",
 "tools/run_kci_frozen_qstar_vs_qpi_cpu.py",
 "tools/kci_train_only_frozen_qpi_teacher_cpu.py",
 "tools/kci_fourway_sft_qstar_qpi_terminal_cpu.py",
 "tools/kci_exact_qpi_prepost_cpu.py",
 "tools/kci_audit_exact_qpi_independent_cpu.py",
 "tools/kci_dg19_downstream_intervention_audit_cpu.py",
 "tools/kci_dg19_source_correction_variants_cpu.py",
 "tools/kci_policy_gap_write_report_cpu.py",
 "tools/kci_package_exact_qpi_fourway_cpu.py",
 "code/tests/test_kci_exact_frozen_qpi_cpu.py",
 "tools/chain_shared_actor_cpu_action_interaction.py",
 "tools/compare_kci_chain_methods_cpu_action_interaction.py",
 "tools/solve_v31_longterm.py",
 "code/kci/upstream_downstream_chain_pilot.py",
 "code/kci/diverse_ci_branch_env.py",
 "code/kci/diverse_ci_long_horizon_env.py",
 "code/kci/diverse_ci_long_horizon_v3.py",
 "code/kci/diverse_ci_long_horizon_v31.py",
)
files={}
for key,folder in F.items():
    for p in folder.rglob("*"):
        if p.is_file():files["evidence/"+key+"/"+p.relative_to(folder).as_posix()]=p
for script in scripts:files["source/"+script]=ROOT/script
saved=ROOT/"artifacts"/"20261120_v31"/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/"SMALL_CPU_CHECKPOINTS"
for p in saved.glob("*.pt"):
    files["FROZEN_ORIGINAL_CHECKPOINTS/"+p.name]=p
assert len([p for p in saved.glob("*.pt")])==6
source=ROOT/"artifacts"/"20261120_v31"/"DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01"/"CHAIN_CAUSAL_PAIRS_EVAL_ONLY.jsonl"
files["ORIGINAL_UNMODIFIED_WORLD_SOURCE/"+source.name]=source
assert all(p.is_file() for p in files.values())
integrity={k:sha256(v.read_bytes()).hexdigest() for k,v in sorted(files.items())}
meta={
 "version":"KCI frozen Qstar vs exact Qpi CPU fourway 2026-11-21",
 "original_28_worlds_sha256":sha256(source.read_bytes()).hexdigest(),
 "reference_policy_checkpoints":6,"actually_trained_CPU_checkpoints":12,
 "Qpi_reference_from_real_policy_softmax_native_successor_states":True,
 "native_Qstar_teacher_not_substituted_for_Qpi":True,
 "native_replay_independently_audited":True,
 "source_world_data_original_unmodified":True,
 "synthetic_rule_based_environment_and_fixed_adapter":True,
 "train_graph_families":4,"dev_graph_families":1,"exploratory_graph_families":2,
 "Qwen_or_other_LLM_weights_trained":False,
 "not_blind_benchmark":True,"no_paid_API_or_GPU":True,
 "file_sha256":integrity,
}
with ZipFile(OUT,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for k,p in sorted(files.items()):
        b=p.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",b):
            raise RuntimeError("Potential API token in "+k)
        if p.stat().st_size>250*1024*1024:
            raise RuntimeError("Oversized source file "+k)
        z.writestr(k,b)
    z.writestr("BUNDLE_SHA256_MANIFEST.json",json.dumps(
        meta,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8"))
with ZipFile(OUT) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    assert all(sha256(z.read(k)).hexdigest()==v for k,v in integrity.items())
receipt={
 "status":"EXACT_QPI_FULL_CPU_RESEARCH_DELIVERABLE_VERIFIED",
 "path":str(OUT),"size_bytes":OUT.stat().st_size,
 "file_count":len(files)+1,"sha256":sha256(OUT.read_bytes()).hexdigest(),
 "reference_snapshots":6,"trained_small_CPU_actor_checkpoints":12,
 "original_world_dataset_modified":False,
 "data_privacy_norms_synthetic":True,
}
(B/"RESEARCH_REPORT"/"PACKAGE_RECEIPT.json").write_text(
 json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
