"""Complete source-backed upstream A P0: exact public planner + five-seed SFT/Qstar.

Contains no actual client's data, secret, Qwen weights or paper claims.
"""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile,ZIP_DEFLATED
import json,re
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/"artifacts"/"20261120_v31"
DEST=B/"KCI_UPSTREAM_A_LONGTERM_VALUE_P0_COST_SFT_VS_CCE.zip"
if DEST.exists():raise FileExistsError(DEST)
P=B/"A_PUBLIC_ONLY_COST_PLANNER_P0_01"
A=B/"A_UPSTREAM_DECISION_LABEL_AND_OBSERVABILITY_AUDIT_01"
S=B/"A_PUBLIC_COST_SFT_CCE_PAIRED_P0_01"
V=B/"A_UPSTREAM_P0_INDEPENDENT_VALIDATION_01"
v=json.loads((V/"VALIDATION.json").read_text(encoding="utf-8"))
assert v["status"]=="INDEPENDENT_UPSTREAM_A_P0_SOURCE_AND_RESULT_AUDIT_PASS"
assert v["counts"]["public_planner_numerical_gap_exact"]==28
source=[
 "tools/audit_upstream_A_sources_and_labels.py",
 "tools/audit_upstream_A_trained_policy_probabilities.py",
 "tools/audit_upstream_A_public_planner.py",
 "tools/train_A_public_cost_sft_vs_cce_p0.py",
 "tools/audit_upstream_A_p0_independent.py",
 "tools/write_upstream_A_p0_review.py",
 "tools/package_upstream_A_p0_audit.py",
 "code/tests/test_kci_upstream_a_p0.py",
 "code/kci/upstream_downstream_chain_pilot.py",
 "code/kci/upstream_downstream_chain_pilot_public_contract.py",
 "tools/chain_shared_actor_cpu_action_interaction.py",
 "tools/compare_kci_chain_methods_cpu_action_interaction.py",
]
files={}
for root_name,folder in [
    ("original_actor_legal_observation_analysis",A),
    ("public_visible_planner_exact_native_Q_gap",P),
    ("small_cpu_A_action_training_with_5_seeds",S),
    ("independent_replay_and_validation",V),
]:
    for f in folder.iterdir():
        if f.is_file():files["evidence/"+root_name+"/"+f.name]=f
for name in source:files["source/"+name]=ROOT/name
doc=ROOT/"docs"/"KCI_UPSTREAM_A_QSTAR_ACTION_BINDING_P0_REVIEW_CN.md"
doc_audit=ROOT/"docs"/"KCI_UPSTREAM_A_QSTAR_ACTION_BINDING_P0_REVIEW_AUDIT.json"
files["report/"+doc.name]=doc
files["report/"+doc_audit.name]=doc_audit
files["frozen_source_snapshot/KCI_SHARED_CPU_ACTOR_RULE_ACTION_INTERACTION_REPAIR.zip"]=(B/"KCI_SHARED_CPU_ACTOR_RULE_ACTION_INTERACTION_REPAIR.zip")
assert all(f.is_file() for f in files.values())
hashes={name:sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
meta={
    "source":"KCI A upstream exact public-planner and label-vs-magnitude P0",
    "train_strict":14,"dev_strict":4,"test_strict":6,"test_tie":2,
    "native_Qstar_gap_public_planner_exact":28,
    "small_CPU_action_SFT_strict_test_correct":[6]*5,
    "small_CPU_Qstar_weighted_SFT_strict_test_correct":[4]*5,
    "small_CPU_Qstar_refDPO_proxy_strict_test_correct":[6]*5,
    "not_a_full_shared_LLM_training_result":True,
    "exploratory_test_not_blind":True,
    "CCE_independent_benefit_demonstrated":False,
    "per_file_sha256":hashes,
}
with ZipFile(DEST,"x",ZIP_DEFLATED,compresslevel=8) as f:
    for name,file in sorted(files.items()):
        raw=file.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise ValueError("Possible API token in "+name)
        f.writestr(name,raw)
    f.writestr("BUNDLE_SHA256_MANIFEST.json",json.dumps(meta,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8"))
with ZipFile(DEST) as f:
    assert f.testzip() is None
    assert len(f.namelist())==len(files)+1
    assert all(sha256(f.read(name)).hexdigest()==value for name,value in hashes.items())
result={
    "status":"VERIFIED_SOURCE_BACKED_A_UPSTREAM_P0_PACKAGE",
    "full_path":str(DEST),"bytes":DEST.stat().st_size,
    "files":len(files)+1,
    "sha256":sha256(DEST.read_bytes()).hexdigest(),
    "model_Qpi_not_tested":True,
    "no_hidden_data_or_API_key_included":True,
}
target=V/"PACKAGE_RECEIPT.json"
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
