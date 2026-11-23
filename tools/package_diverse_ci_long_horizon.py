r"""Bundle truly different knowledge-rule and business-branch KCI pilots.

Both original structural pilot and enhanced preflight variant are preserved.
All actor observations are isolated from evaluator-only labels. Archives include
source, exact corpus, independent audits, real CPU baseline, and MIT license.
"""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile,ZIP_DEFLATED
import json,re

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261119_composition_signals"
V1=BASE/"diverse_ci_branch_v1"
V2=BASE/"diverse_ci_long_horizon_v2"
DEST=BASE/"KCI_DIVERSE_KNOWLEDGE_BRANCHES_LONG_HORIZON_V2.zip"
if DEST.exists():raise FileExistsError(DEST)
assert json.loads((V2/"INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))["status"]=="FULL_INDEPENDENT_NATIVE_REPLAY_PASS"
val=json.loads((V2/"VALIDATION.json").read_text(encoding="utf-8"))
assert val["generated_structural_roots"]==21
assert val["real_worlds"]==189
assert val["certified_immediate_action_labels"]==175
assert val["model_action_observations"]==539
base=json.loads((V2/"CPU_POLICY_BASELINES.json").read_text(encoding="utf-8"))
assert (base["result"]["myopic_skip_success"],
        base["result"]["myopic_always_open_success"],
        base["result"]["role_conditioned_preflight_success"],
        base["result"]["oracle_success"])==(28,56,63,63)
for folder in (V1,V2):
    manifest=json.loads((folder/"MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    for name,h in manifest.items():
        assert sha256((folder/name).read_bytes()).hexdigest()==h,(folder,name)
scripts=[
    "code/kci/diverse_ci_branch_env.py",
    "code/kci/diverse_ci_long_horizon_env.py",
    "code/tests/test_diverse_ci_branch_env.py",
    "code/tests/test_diverse_ci_long_horizon_env.py",
    "tools/pilot_diverse_ci_branches.py",
    "tools/baseline_diverse_ci_branch_cpu.py",
    "tools/build_diverse_ci_branch_dataset.py",
    "tools/pilot_diverse_ci_long_horizon.py",
    "tools/build_diverse_ci_long_horizon_dataset.py",
    "tools/audit_diverse_ci_long_horizon_dataset.py",
    "tools/baseline_diverse_ci_long_horizon_cpu.py",
]
files={
    **{"datasets/long_horizon_v2/"+p.name:p for p in V2.iterdir() if p.is_file()},
    **{"datasets/structural_only_v1/"+p.name:p for p in V1.iterdir() if p.is_file()},
    **{rel:ROOT/rel for rel in scripts},
    "THIRD_PARTY_LICENSES/AGENTDOJO_MIT_LICENSE.txt":
        Path(r"D:\research\kci_public_refs_20261119\agentdojo\LICENSE"),
    "THIRD_PARTY_LICENSES/TAU2_MIT_LICENSE.txt":
        Path(r"D:\research\kci_public_refs_20261119\tau2-bench\LICENSE"),
}
assert all(p.is_file() for p in files.values())
checksum={name:sha256(p.read_bytes()).hexdigest() for name,p in sorted(files.items())}
with ZipFile(DEST,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,path in sorted(files.items()):
        raw=path.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise RuntimeError("Possible credential in "+name)
        z.writestr(name,raw)
    z.writestr("BUNDLE_SHA256_MANIFEST.json",
        json.dumps({
            "name":"KCI Different Knowledge DAGs + Dynamic Multiagent Business Paths",
            "version":"long_horizon_v2",
            "AI_model_results_present":False,
            "official_external_benchmark":False,
            "human_norm_certification":False,
            "root_count":21,"knowledge_graphs":7,"workflow_types":3,
            "worlds":189,"native_model_inputs":539,
            "context_pairs":105,"native_execution_traces":168,
            "files":checksum,
        },ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8"))
with ZipFile(DEST,"r") as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    assert all(sha256(z.read(name)).hexdigest()==h for name,h in checksum.items())
result={
    "status":"FINAL_ZIP_VERIFIED",
    "file":str(DEST),
    "bytes":DEST.stat().st_size,
    "SHA256":sha256(DEST.read_bytes()).hexdigest(),
    "entries":len(files)+1,
    "new_knowledge_graphs":7,
    "new_structural_task_roots":21,
    "worlds":189,
    "model_observations":539,
    "historical_business_pairs":105,
    "full_executable_traces":168,
    "new_vs_old_cpu_myopic_baseline":[56,28],
    "old_scenarios_count":56,"new_scenarios_count":63,
    "no_real_enterprise_data":True,
    "no_secret_in_zip":True,
    "model_training_completed":False}
receipt=V2/"PACKAGE_RECEIPT.json"
if receipt.exists():raise FileExistsError(receipt)
receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
