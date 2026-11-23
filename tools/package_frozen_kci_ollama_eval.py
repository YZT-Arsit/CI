"""Package actual local frozen-model action experiments and independent native replay.

No model weights, API keys, or third-party code. Model ID/digest suffices for
reloading the user's existing locally installed Ollama checkpoint.
"""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from hashlib import sha256
import json,re
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261119_composition_signals"
P=BASE/"frozen_qwen25_7b_diverse_probes_01"
F=BASE/"frozen_qwen25_7b_diverse_full_01"
B=BASE/"frozen_qwen25_7b_order_bias_01"
REPORT=BASE/"FROZEN_QWEN25_7B_DIVERSE_EVALUATION_01"
DATA=BASE/"diverse_ci_long_horizon_v2"
ZIP=BASE/"KCI_FROZEN_QWEN25_7B_DIVERSE_EVAL.zip"
if ZIP.exists():raise FileExistsError(ZIP)
for p in (P,F):
    audit=json.loads((p/"INDEPENDENT_REPLAY_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["status"]=="INDEPENDENT_FROZEN_POLICY_NATIVE_REPLAY_PASS"
assert json.loads((DATA/"INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))["status"]=="FULL_INDEPENDENT_NATIVE_REPLAY_PASS"
assert json.loads((B/"ORDER_BIAS.json").read_text(encoding="utf-8"))["status"]=="ORDER_SENSITIVITY_EVALUATED"
assert json.loads((REPORT/"CONSOLIDATED_RESULTS.json").read_text(encoding="utf-8"))["no_training"] is True
sources=[
    ROOT/"code"/"kci"/"diverse_ci_branch_env.py",
    ROOT/"code"/"kci"/"diverse_ci_long_horizon_env.py",
    ROOT/"tools"/"pilot_diverse_ci_long_horizon.py",
    ROOT/"tools"/"evaluate_frozen_kci_ollama.py",
    ROOT/"tools"/"audit_frozen_kci_ollama_results.py",
    ROOT/"tools"/"check_frozen_kci_action_order_bias.py",
    ROOT/"tools"/"summarize_frozen_diverse_ci_eval.py",
]
files={
    **{"frozen_probes/"+path.name:path for path in P.iterdir() if path.is_file()},
    **{"frozen_full_episodes/"+path.name:path for path in F.iterdir() if path.is_file()},
    **{"action_order_control/"+path.name:path for path in B.iterdir() if path.is_file()},
    **{"analysis/"+path.name:path for path in REPORT.iterdir() if path.is_file()},
    **{"development_environment/"+path.name:DATA/path.name for path in (
        DATA/"VALIDATION.json",DATA/"INDEPENDENT_AUDIT.json",DATA/"README_CN.md",
        DATA/"CPU_POLICY_BASELINES.json",DATA/"ROOTS.jsonl",
        DATA/"KNOWLEDGE_GRAPHS.jsonl")},
    **{path.relative_to(ROOT).as_posix():path for path in sources},
}
assert all(path.exists() for path in files.values())
hashes={name:sha256(p.read_bytes()).hexdigest() for name,p in sorted(files.items())}
with ZipFile(ZIP,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,path in sorted(files.items()):
        contents=path.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",contents):
            raise RuntimeError("Potential credential in "+name)
        z.writestr(name,contents)
    z.writestr("BUNDLE_SHA256_MANIFEST.json",json.dumps({
        "study":"Actual frozen local Ollama Qwen2.5-7B q4 native agent evaluation",
        "not_the_remote_Qwen3_4B":True,
        "no_training":True,
        "no_paid_API_calls":True,
        "evaluation_not_blind":True,
        "synthetic_business_and_CI_norms":True,
        "files":hashes
    },ensure_ascii=False,indent=2,sort_keys=True).encode("utf-8"))
with ZipFile(ZIP) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    assert all(sha256(z.read(name)).hexdigest()==h for name,h in hashes.items())
result={
    "status":"ZIP_VERIFIED",
    "path":str(ZIP),
    "bytes":ZIP.stat().st_size,
    "sha256":sha256(ZIP.read_bytes()).hexdigest(),
    "entries":len(files)+1,
    "local_model_name":"qwen2.5:7b-instruct-q4_K_M",
    "full_end_to_end_native_replay_audit":True,
    "no_private_keys_or_model_weights_in_bundle":True
}
(REPORT/"PACKAGE_RECEIPT.json").write_text(
    json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
