r"""Create self-contained 10-task synthetic KCI grounded workflow archive."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from hashlib import sha256
import json,re
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261119_composition_signals"
DATA=BASE/"public_workflow_all10_grounded_v1"
TWO=BASE/"public_workflow_grounded_W01_R03_v1"
EIGHT=BASE/"public_workflow_remaining_8_grounded_v1"
OUT=BASE/"KCI_ALL10_GROUNDED_WORKFLOWS_V1.zip"
if OUT.exists():raise FileExistsError(OUT)
assert json.loads((DATA/"INDEPENDENT_CROSS_FAMILY_AUDIT.json").read_text(
    encoding="utf-8"))["result"]=="CROSS_FAMILY_AUDIT_PASS"
assert json.loads((EIGHT/"INDEPENDENT_AUDIT.json").read_text(
    encoding="utf-8"))["result"]=="INDEPENDENT_REPLAY_PASS"
assert json.loads((TWO/"INDEPENDENT_AUDIT.json").read_text(
    encoding="utf-8"))["status"]=="INDEPENDENT_CPU_AUDIT_PASS"
for folder in (DATA,TWO,EIGHT):
    manifest=json.loads((folder/"MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    for name,expected in manifest.items():
        assert sha256((folder/name).read_bytes()).hexdigest()==expected,(folder,name)
code=[
    ROOT/"code"/"kci"/"knowledge_world_seed_v2.py",
    ROOT/"code"/"kci"/"public_workflow_ci.py",
    ROOT/"code"/"kci"/"public_workflow_ci_grounded.py",
    ROOT/"code"/"kci"/"public_workflow_ci_remaining_grounded.py",
    ROOT/"code"/"tests"/"test_public_workflow_ci.py",
    ROOT/"code"/"tests"/"test_public_workflow_ci_grounded.py",
    ROOT/"code"/"tests"/"test_public_workflow_ci_remaining_grounded.py",
    ROOT/"code"/"tests"/"test_knowledge_world_seed_v2.py",
    ROOT/"tools"/"build_public_ci_grounded_two.py",
    ROOT/"tools"/"audit_public_ci_grounded_two.py",
    ROOT/"tools"/"build_public_ci_remaining_grounded.py",
    ROOT/"tools"/"audit_public_ci_remaining_grounded.py",
    ROOT/"tools"/"assemble_public_ci_all10_grounded.py",
    ROOT/"tools"/"audit_public_ci_all10_grounded.py",
]
sources={
    "THIRD_PARTY_LICENSES/AGENTDOJO_MIT_LICENSE.txt":
       Path(r"D:\research\kci_public_refs_20261119\agentdojo\LICENSE"),
    "THIRD_PARTY_LICENSES/TAU2_MIT_LICENSE.txt":
       Path(r"D:\research\kci_public_refs_20261119\tau2-bench\LICENSE"),
}
files={
    **{f"datasets/all10/{f.name}":f for f in DATA.iterdir() if f.is_file()},
    **{f"datasets/w01_r03/{f.name}":f for f in TWO.iterdir() if f.is_file()},
    **{f"datasets/new_eight/{f.name}":f for f in EIGHT.iterdir() if f.is_file()},
    **{p.relative_to(ROOT).as_posix():p for p in code},
    **sources,
}
assert all(p.is_file() for p in files.values())
checksums={name:sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())}
with ZipFile(OUT,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,path in sorted(files.items()):
        raw=path.read_bytes()
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise RuntimeError("Possible credential content in "+name)
        z.writestr(name,raw)
    z.writestr("PACKAGE_SHA256_MANIFEST.json",
        json.dumps({
            "study":"KCI ALL10 GROUNDED WORKFLOWS V1",
            "training_ready":False,
            "source_task_roots":10,
            "world_scenarios":80,
            "controlled_contrasts":30,
            "real_simulated_full_trajectories":60,
            "negative_controls":24,
            "files":checksums,
        },ensure_ascii=False,indent=2,sort_keys=True).encode())
with ZipFile(OUT) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    assert all(sha256(z.read(name)).hexdigest()==h for name,h in checksums.items())
receipt={
    "status":"ZIP_INTEGRITY_VERIFIED",
    "path":str(OUT),"size_bytes":OUT.stat().st_size,
    "sha256":sha256(OUT.read_bytes()).hexdigest(),
    "files":len(files)+1,
    "task_roots":10,"worlds":80,
    "controlled_pairs":30,"verified_full_native_business_traces":60,
    "negative_controls":24,
    "official_upstream_harness_used":False,
    "human_social_norm_certified":False,
    "all_labels_synthetic":True,
    "API_key_embedded":False,
    "GPU_training_completed":False
}
f=DATA/"PACKAGE_RECEIPT.json"
if f.exists():raise FileExistsError(f)
f.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
