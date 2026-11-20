"""Package verified KCI V3 P0 history-conditioned Qstar flips and P1 branches.

Source files, real native continuations, 735 real native heuristic policies,
anti-shortcut controls, independent replay, and full SHA256 checks.
No secrets/model weights; output lives under ignored research artifacts until
a separate explicit Git commit selects it.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile,ZIP_DEFLATED
import json,re
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261119_composition_signals"
MAIN=BASE/"KCI_V3_TRUE_FLIPS_DATASET_01"
P0=BASE/"P0_TRUE_LONG_HORIZON_FLIP_V3_ENTITY_MATCHED"
P1=BASE/"KCI_V3_BRANCH_CONTROLS_01"
TARGET=BASE/"KCI_V3_VERIFIED_LONGTERM_HISTORY_FLIPS_FINAL.zip"
if TARGET.exists():raise FileExistsError(TARGET)
assert json.loads((MAIN/"INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))["status"]=="INDEPENDENT_EXACT_NATIVE_QSTAR_AND_DATA_LEAKAGE_AUDIT_PASS"
assert json.loads((P1/"VALIDATION.json").read_text(encoding="utf-8"))["status"]=="V3_TRUE_BUSINESS_BRANCHES_AND_BROKER_CHECK_PASS"
assert json.loads((P0/"VALIDATION.json").read_text(encoding="utf-8"))["strict_preflight_action_value_flips"]==35
assert json.loads((MAIN/"VALIDATION.json").read_text(encoding="utf-8"))["total_strict_related_vs_unrelated_pairs"]==70
assert json.loads((MAIN/"CPU_ACTUAL_POLICY_BASELINES.json").read_text(encoding="utf-8"))["status"]=="REAL_NATIVE_FULL_POLICY_REPLAY_COMPLETE"
for source in (MAIN,P0,P1):
    file=source/"MANIFEST_SHA256.json"
    if not file.exists():
        if source==P0:
            data=json.loads((P0/"VALIDATION.json").read_text(encoding="utf-8"))
            assert data["file_sha256"]==sha256((P0/"P0_EXACT_BRANCHES.jsonl").read_bytes()).hexdigest()
            continue
        raise FileNotFoundError(file)
    m=json.loads(file.read_text(encoding="utf-8"))
    for name,expected in m.items():
        assert sha256((source/name).read_bytes()).hexdigest()==expected,(source,name)
code=[
    "code/kci/diverse_ci_branch_env.py",
    "code/kci/diverse_ci_long_horizon_env.py",
    "code/kci/diverse_ci_long_horizon_v3.py",
    "code/tests/test_true_longterm_flip_v3.py",
    "tools/solve_longterm_flip_v3.py",
    "tools/scan_p0_long_horizon_flips_v3.py",
    "tools/build_kci_v3_longterm_dataset.py",
    "tools/audit_kci_v3_longterm_dataset.py",
    "tools/run_v3_native_strong_baselines.py",
    "tools/audit_v3_business_routes.py",
]
files={
    **{"dataset/main/"+p.name:p for p in MAIN.iterdir() if p.is_file() and not p.name.startswith("PACKAGE_RECEIPT")},
    **{"dataset/P0_exact_history_flips/"+p.name:p for p in P0.iterdir() if p.is_file()},
    **{"dataset/P1_business_routes/"+p.name:p for p in P1.iterdir() if p.is_file()},
    **{p:str(ROOT/p) for p in code},
}
source_files={name:Path(path) for name,path in files.items()}
assert all(p.is_file() for p in source_files.values())
hashes={key:sha256(p.read_bytes()).hexdigest() for key,p in sorted(source_files.items())}
manifest={
    "name":"KCI V3 Verified Long-term Receiver Knowledge History Flips",
    "version":"diverse_ci_long_horizon_v3_verified_flip",
    "actual_native_Qstar_paired_branch_worlds":210,
    "fully_executed_native_oracle_continuations":420,
    "strict_longterm_P0_flips":70,
    "independent_task_graph_business_conditions":35,
    "strong_actual_CPU_policy_episodes":735,
    "P1_distinct_business_branch_controls":28,
    "balanced_train_only_oracle_preferences":40,
    "trained_LLM_policy_Qpi_available":False,
    "not_a_human_norm_certified_or_official_upstream_benchmark":True,
    "full_bundle_files":hashes,
}
with ZipFile(TARGET,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,path in sorted(source_files.items()):
        raw=path.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise RuntimeError("Possible API token in: "+name)
        z.writestr(name,raw)
    z.writestr("BUNDLE_SHA256_MANIFEST.json",
               json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8"))
with ZipFile(TARGET) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(source_files)+1
    assert all(sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
r={
    "status":"VERIFIED_ARCHIVE",
    "path":str(TARGET),"size_bytes":TARGET.stat().st_size,
    "sha256":sha256(TARGET.read_bytes()).hexdigest(),
    "file_count":len(source_files)+1,
    "core_P0_flips":70,
    "native_two_action_continuations":420,
    "strong_baseline_native_episodes":735,
    "P1_branch_controls":28,
    "API_keys_included":False,
    "not_a_trained_model":True,
}
p=MAIN/"PACKAGE_RECEIPT_FINAL.json"
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)
