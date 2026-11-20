r"""Bundle source-backed Gate B *preliminary* CPU trainability and frozen LLM checks.

Includes checkpoints for the 12 small PyTorch learners, 60 actual frozen
Qwen2.5-7B action outputs and independent replay, terminal bandit 800-epoch
robustness, original synthetic Gate A dataset, and exact scripts.

Does NOT include Qwen weights, any API credential or hidden private source.
"""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from hashlib import sha256
import json,re
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261120_v31"
OUT=BASE/"KCI_V31_SMALL_SCALE_CPU_GATE_B_PILOT.zip"
if OUT.exists():raise FileExistsError(OUT)
folders={
    "report":BASE/"KCI_V31_CPU_GATE_B_SMALL_SCALE_REPORT",
    "neural_160_epochs":BASE/"CPU_GATE_B_NEURAL_01",
    "neural_checkpoint_audit":BASE/"CPU_GATE_B_NEURAL_REPRODUCIBLE_03",
    "terminal_800_epochs":BASE/"CPU_GATE_B_TERMINAL_BANDIT_800_EPOCHS",
    "frozen_local_qwen25":BASE/"FROZEN_QWEN25_V31_PRELFIGHT_01",
}
report=json.loads((folders["report"]/"AUDIT_SUMMARY.json").read_text(encoding="utf-8"))
assert report["status"]=="SOURCE_VERIFIED_GATE_B_CPU_PRELIMINARY"
assert report["value_rule_test_accuracy"]==1.0
assert report["terminal_bandit_800_test_accuracy"]==1.0
assert report["frozen_Qwen_inference_calls"]==60
files={}
for category,path in folders.items():
    for p in path.rglob("*"):
        if p.is_file():
            files[category+"/"+p.relative_to(path).as_posix()]=p
source=[
    "tools/train_v31_cpu_policy_probe.py",
    "tools/audit_cpu_policy_gateb_permutations.py",
    "tools/audit_terminal_bandit_long_budget_v31.py",
    "tools/evaluate_v31_frozen_ollama_fewshot.py",
    "tools/audit_v31_frozen_qwen_icl.py",
    "tools/write_v31_gateb_cpu_report.py",
    "tools/run_v31_shortcut_learning_gate.py",
    "tools/audit_kci_v31_gateA.py",
    "tools/solve_v31_longterm.py",
    "code/kci/diverse_ci_long_horizon_v31.py",
]
for name in source:files["source/"+name]=ROOT/name
files["grounded_source_dataset/KCI_V31_GATE_A_SHORTCUT_FIXED_DEVELOPMENT.zip"]=BASE/"KCI_V31_GATE_A_SHORTCUT_FIXED_DEVELOPMENT.zip"
assert all(path.exists() and path.is_file() for path in files.values())
manifest={name:sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
with ZipFile(OUT,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,path in sorted(files.items()):
        raw=path.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise RuntimeError("Potential secret string in "+name)
        if path.stat().st_size>20*1024*1024:
            raise RuntimeError("Individual file too large "+name)
        z.writestr(name,raw)
    z.writestr("BUNDLE_SHA256_MANIFEST.json",
        json.dumps({
            "study":"KCI V3.1 small CPU parameter-training and frozen Qwen evaluation",
            "not_a_full_Qwen3_4B_training_experiment":True,
            "Gate_B_outcome":"CPU SURROGATE ONLY; actual LLM trainability not proven",
            "trainable_neural_horn_parameters":4,
            "small_Torch_models_trained":12,
            "heldout_graph_test_worlds":36,
            "frozen_Qwen2_5_7B_actions":60,
            "original_Qwen3_4B_weights_included":False,
            "API_credentials_included":False,
            "source_hashes":manifest,
        },ensure_ascii=False,indent=2,sort_keys=True).encode("utf-8"))
with ZipFile(OUT) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    assert all(sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
receipt={
    "status":"GATE_B_CPU_PILOT_PACKAGE_VERIFIED",
    "file":str(OUT),
    "bytes":OUT.stat().st_size,
    "sha256":sha256(OUT.read_bytes()).hexdigest(),
    "file_count":len(files)+1,
    "actual_frozen_actions":60,
    "small_CPU_neural_weights":12,
    "not_Qwen3_4B_train":True,
}
( folders["report"]/"PACKAGE_RECEIPT.json").write_text(
    json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
