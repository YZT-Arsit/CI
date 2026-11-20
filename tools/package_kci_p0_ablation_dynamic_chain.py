r"""Standalone snapshot of decisive BCE vs Q*-margin and actual A/R/B/C pilot.

Includes all original CPU native transcripts, five-seed short/low-budget
learning curves, independent dynamic chain replay, standalone source,
tests, and the frozen V3.1 Gate-A dataset archive. Does NOT contain any
trained model weights or personalized/private business records.
"""
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from hashlib import sha256
import json,re
R=Path(__file__).resolve().parents[1]
BASE=R/"artifacts"/"20261120_v31"
TARGET=BASE/"KCI_P0_ACTION_LABEL_ABLATION_AND_DYNAMIC_CHAIN_PILOT.zip"
if TARGET.exists():raise FileExistsError(TARGET)
dirs={
    "action_label_vs_margin":BASE/"P0_LABEL_VS_QMARGIN_BCE_20261120",
    "low_data_action_label_vs_margin":BASE/"P0_LOW_DATA_LABEL_VS_QMARGIN_ABLATION",
    "upstream_dynamic_chain":BASE/"DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01",
}
assert json.loads((dirs["action_label_vs_margin"]/"ABLATION_RESULTS.json").read_text(encoding="utf-8"))["status"]=="ACTUAL_MATCHED_CPU_ABLATION_COMPLETED"
assert json.loads((dirs["low_data_action_label_vs_margin"]/"RESULTS.json").read_text(encoding="utf-8"))["status"]=="LOW_DATA_PAIRED_ACTION_LABEL_VS_QMARGIN_COMPLETED"
assert json.loads((dirs["upstream_dynamic_chain"]/"INDEPENDENT_CAUSAL_REPLAY_AUDIT.json").read_text(encoding="utf-8"))["status"]=="INDEPENDENT_DYNAMIC_CHAIN_CAUSAL_ACTION_REPLAY_PASS"
manifest=json.loads((dirs["upstream_dynamic_chain"]/"MANIFEST_SHA256.json").read_text(encoding="utf-8"))
for name,h in manifest.items():
    assert sha256((dirs["upstream_dynamic_chain"]/name).read_bytes()).hexdigest()==h,name
source=[
    "code/kci/upstream_downstream_chain_pilot.py",
    "code/kci/diverse_ci_branch_env.py",
    "code/kci/diverse_ci_long_horizon_env.py",
    "code/kci/diverse_ci_long_horizon_v3.py",
    "code/kci/diverse_ci_long_horizon_v31.py",
    "code/tests/test_kci_dynamic_chain_pilot.py",
    "tools/solve_v31_longterm.py",
    "tools/train_v31_cpu_policy_probe.py",
    "tools/run_v31_shortcut_learning_gate.py",
    "tools/decisive_v31_label_vs_qmargin_ablation.py",
    "tools/ablate_v31_low_data_q_weights.py",
    "tools/build_kci_dynamic_chain_pilot.py",
    "tools/audit_kci_dynamic_chain_pilot.py",
    "tools/write_kci_decisive_ablation_and_chain_report.py",
]
files={}
for folder,p in dirs.items():
    for f in p.iterdir():
        if f.is_file():files["results/"+folder+"/"+f.name]=f
files.update({"source/"+name:R/name for name in source})
files["report/KCI_P0_LABEL_VS_QSTAR_AND_DYNAMIC_AGENT_CHAIN_CN.md"]=R/"docs"/"KCI_P0_LABEL_VS_QSTAR_AND_DYNAMIC_AGENT_CHAIN_CN.md"
files["report/KCI_P0_LABEL_VS_QSTAR_AND_DYNAMIC_AGENT_CHAIN_CN.audit.json"]=R/"docs"/"KCI_P0_LABEL_VS_QSTAR_AND_DYNAMIC_AGENT_CHAIN_CN.audit.json"
files["source_dataset/KCI_V31_GATE_A_SHORTCUT_FIXED_DEVELOPMENT.zip"]=BASE/"KCI_V31_GATE_A_SHORTCUT_FIXED_DEVELOPMENT.zip"
assert all(x.is_file() for x in files.values())
full={name:sha256(f.read_bytes()).hexdigest() for name,f in sorted(files.items())}
with ZipFile(TARGET,"x",ZIP_DEFLATED,compresslevel=8) as archive:
    for name,sourcefile in sorted(files.items()):
        raw=sourcefile.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise RuntimeError("Secret-like content "+name)
        if re.search(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",raw):
            raise RuntimeError("Private-key-like bytes "+name)
        archive.writestr(name,raw)
    archive.writestr("BUNDLE_SHA256_MANIFEST.json",json.dumps({
        "study":"KCI real native A->R->B->C chain and decisive optimal-label-vs-Qstar-weight ablation",
        "no_Qwen_training":True,
        "original_V31_archives_not_modified":True,
        "source_CI_norms_fictional":True,
        "not_ICML_method_proof":True,
        "five_seed_matched_label_ablation":True,
        "native_dynamic_chain_112_trajectories":True,
        "files_sha256":full
    },ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8"))
with ZipFile(TARGET) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist())==len(files)+1
    for name,h in full.items():
        assert sha256(archive.read(name)).hexdigest()==h
receipt={
    "status":"BUNDLE_INTEGRITY_SHA256_VERIFIED",
    "path":str(TARGET),
    "bytes":TARGET.stat().st_size,
    "sha256":sha256(TARGET.read_bytes()).hexdigest(),
    "files":len(files)+1,
    "source_frozen_V31_datasets_present":True,
    "Qpi_LLMAgent_training_done":False,
}
RCP=dirs["upstream_dynamic_chain"]/"BUNDLE_RECEIPT.json"
if RCP.exists():raise FileExistsError(RCP)
RCP.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
