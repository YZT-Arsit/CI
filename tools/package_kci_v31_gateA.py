r"""Deliver standalone V3.1 Gate-A synthetic data, controls and source.

Selected ZIP is safe to commit: no credentials, no model files and no private
raw business data. Includes actual native Qstar evidence and real native
baseline step paths. Does not claim paper-grade blind test or model Qpi.
"""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from hashlib import sha256
import re,json
R=Path(r"D:\research\longhistory")
B=R/"artifacts"/"20261120_v31"
D=B/"V31_GATEA_FINAL_CANDIDATE"
P=B/"V31_BROKER_MULTIROle_P1_02"
OUT=B/"KCI_V31_GATE_A_SHORTCUT_FIXED_DEVELOPMENT.zip"
if OUT.exists():raise FileExistsError(OUT)
assert json.loads((D/"GATE_A_INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))["status"]=="GATE_A_INDEPENDENT_REPLAY_PASS"
assert json.loads((P/"VALIDATION.json").read_text(encoding="utf-8"))["status"]=="V31_P1_ACTUAL_BROKER_RECIPIENT_HANDBRANCH_PASS"
assert json.loads((D/"GATE_A_STRONG_SHORTCUT_AND_NATIVE_ROLLOUTS.json").read_text(encoding="utf-8"))["status"]=="V31_BALANCED_GATE_A_STRONG_BASELINES_EXECUTED"
assert json.loads((D/"TRAIN_MULTIROLE_AUDIT.json").read_text(encoding="utf-8"))["status"]=="TRAIN_ONLY_NATIVE_MULTIROLE_REFERENCE_READY"
for folder in (D,P):
    m=json.loads((folder/"MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    for name,h in m.items():
        assert sha256((folder/name).read_bytes()).hexdigest()==h,(str(folder),name)
    # The source-generation manifest explicitly does not cover the later
    # independent audit additions: this ZIP adds a full bundle manifest.
files={
    **{"data/v31/"+x.name:x for x in D.iterdir() if x.is_file()},
    **{"data/P1_real_broker/"+x.name:x for x in P.iterdir() if x.is_file()},
}
src=[
    "code/kci/diverse_ci_branch_env.py",
    "code/kci/diverse_ci_long_horizon_env.py",
    "code/kci/diverse_ci_long_horizon_v3.py",
    "code/kci/diverse_ci_long_horizon_v31.py",
    "code/tests/test_kci_v31_gateA.py",
    "tools/solve_v31_longterm.py",
    "tools/build_kci_v31_pilot.py",
    "tools/audit_kci_v31_gateA.py",
    "tools/run_v31_shortcut_learning_gate.py",
    "tools/derive_v31_train_multirole.py",
    "tools/audit_kci_v31_multirole_relay.py",
]
files.update({name:R/name for name in src})
assert all(x.is_file() for x in files.values())
sha={name:sha256(x.read_bytes()).hexdigest() for name,x in sorted(files.items())}
meta={
    "status":"GATE_A_SYNTHETIC_DEVELOPMENT_DATA_NOT_LEARNED_MODEL",
    "case_mechanisms":63,"primary_balanced_worlds":126,
    "worlds":315,"exact_native_two_action_continuations":630,
    "native_oracle_role_observations":2030,
    "train_graph_native_multi_role_imitation_samples":485,
    "real_broker_control_roots":7,
    "Qstar_not_Qpi":True,
    "human_social_norm_certified":False,
    "official_upstream_benchmark":False,
    "true_blind_test":False,
    "no_personal_data_or_API_key":True,
    "files_sha256":sha,
}
with ZipFile(OUT,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,x in sorted(files.items()):
        raw=x.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise RuntimeError("Secret-like key content in "+name)
        key_marker=b"-----BEGIN "+b"OPENSSH "+b"PRIVATE KEY-----"
        if key_marker in raw:
            raise RuntimeError("Private key present in "+name)
        z.writestr(name,raw)
    z.writestr("BUNDLE_CONTENTS_SHA256.json",
        json.dumps(meta,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8"))
with ZipFile(OUT) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    assert all(sha256(z.read(name)).hexdigest()==h for name,h in sha.items())
print(json.dumps({
    "status":"ARCHIVE_VALIDATED",
    "path":str(OUT),"size_bytes":OUT.stat().st_size,
    "sha256":sha256(OUT.read_bytes()).hexdigest(),
    "files":len(files)+1,"model_trained":False,
    "Gate_A":True
},ensure_ascii=False,indent=2),flush=True)
