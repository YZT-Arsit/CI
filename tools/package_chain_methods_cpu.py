r"""Bundle actual trained shared CPU role policy comparisons with independent replay.

Does not contain API credentials, model weights from Qwen, or actual client data.
Dataset is synthetic and developer-inspected. CCE/DPR are native surrogates.
"""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile,ZIP_DEFLATED
import json,re
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261120_v31"
EXP=BASE/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_01"
REF=BASE/"CHAIN_NATIVE_ORACLE_BASELINES_01"
DPR2=BASE/"CHAIN_DPR_BACKTRACKED_POLICY_COMPARISON_02"
SOURCE=BASE/"DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01"
DEST=BASE/"KCI_SHARED_CPU_CCE_DPR_ACTION_SFT_TERMINAL_COMPARISON.zip"
if DEST.exists():raise FileExistsError(DEST)
audit=json.loads((EXP/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json").read_text(encoding="utf-8"))
assert audit["status"]=="INDEPENDENT_SHARED_CPU_ACTOR_NATIVE_REPLAY_PASS"
result=json.loads((EXP/"RESULTS.json").read_text(encoding="utf-8"))
assert result["status"]=="CPU_SHARED_ROLE_POLICY_METHOD_COMPARISON_REAL_NATIVE_ROLLOUTS"
assert len(result["per_seed_and_heldout_full_native_model_action_paths"])==15
ref=json.loads((REF/"SUMMARY.json").read_text(encoding="utf-8"))
assert ref["status"]=="EXACT_NATIVE_PILOT_REFERENCE_CEILING_VERIFIED"
dpr2_audit=json.loads((DPR2/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json").read_text(encoding="utf-8"))
assert dpr2_audit["status"]=="INDEPENDENT_BACKTRACKED_DPR_NATIVE_REPLAY_PASS"
files={}
for title,folder in (
    ("experiment_actual_CPU_policy",EXP),
    ("exact_native_reference",REF),
    ("backtracked_DPR_native_replay",DPR2),
    ("original_dynamic_synthetic_source",SOURCE)):
    for f in folder.iterdir():
        if f.is_file():files[title+"/"+f.name]=f
code=(
    "tools/chain_shared_actor_cpu.py",
    "tools/compare_kci_chain_methods_cpu.py",
    "tools/audit_chain_methods_cpu.py",
    "tools/baseline_chain_native_planner_cpu.py",
    "tools/write_chain_cpu_method_comparison_report.py",
    "tools/chain_dpr_backtrack_v2.py",
    "tools/compare_chain_dpr_backtracked_cpu.py",
    "tools/audit_chain_dpr_backtracked_cpu.py",
    "code/tests/test_chain_shared_cpu_comparison.py",
    "code/kci/upstream_downstream_chain_pilot.py",
    "code/kci/diverse_ci_branch_env.py",
    "code/kci/diverse_ci_long_horizon_env.py",
    "code/kci/diverse_ci_long_horizon_v3.py",
    "code/kci/diverse_ci_long_horizon_v31.py",
    "tools/solve_v31_longterm.py",
)
for path in code:
    files["source/"+path]=ROOT/path
DOC=ROOT/"docs"/"KCI_SHARED_CPU_CHAIN_METHODS_COMPARISON_CN.md"
files["report/"+DOC.name]=DOC
AUDIT_DOC=DOC.with_suffix(".audit.json")
files["report/"+AUDIT_DOC.name]=AUDIT_DOC
files["previous_frozen_mechanism_evidence/KCI_P0_ACTION_LABEL_ABLATION_AND_DYNAMIC_CHAIN_PILOT.zip"]=BASE/"KCI_P0_ACTION_LABEL_ABLATION_AND_DYNAMIC_CHAIN_PILOT.zip"
assert all(p.is_file() for p in files.values())
hashes={name:sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
meta={
    "study":"CPU shared native role-conditioned A->R->B->C Action SFT vs Terminal REINFORCE vs native CCE/DPR proxies",
    "trained_model":"4.5k param SharedHornActor; NOT Qwen3-4B or learned LLM",
    "training_graphs":4,"development_graphs":1,"exploratory_test_graphs":2,
    "shared_initialization_seed_per_method":True,
    "not_a_true_blind_test":True,
    "not_full_original_CCE_DPR_algorithms":True,
    "improved_DPR_backtrack_verified_pairs_per_seed":[4,4,6],
    "same_graph_test_CCE_and_DPR_incremental_benefit_observed":False,
    "all_synthetic_environment":True,
    "per_file_sha256":hashes,
}
with ZipFile(DEST,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,p in sorted(files.items()):
        raw=p.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise RuntimeError("Possible secret in "+name)
        z.writestr(name,raw)
    z.writestr("BUNDLE_SHA256_MANIFEST.json",json.dumps(
        meta,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8"))
with ZipFile(DEST) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    assert all(sha256(z.read(name)).hexdigest()==h for name,h in hashes.items())
receipt={
    "status":"ZIP_CONTENT_AND_SHA256_VERIFIED",
    "filename":DEST.name,"full_path":str(DEST),
    "bytes":DEST.stat().st_size,"files":len(files)+1,
    "sha256":sha256(DEST.read_bytes()).hexdigest(),
    "Qwen_model_training":False,
    "native_path_replay_audited":True,
}
path=EXP/"PACKAGE_RECEIPT.json"
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
