r"""Package the independently validated KCI public-workflow pilot without API keys."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile,ZIP_DEFLATED
import json
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"artifacts"/"20261119_composition_signals"/"public_workflow_ci_10_final"
TARGET=ROOT/"artifacts"/"20261119_composition_signals"/"KCI_PUBLIC_WORKFLOW_CI_10_PILOT.zip"
if TARGET.exists():raise FileExistsError(TARGET)
required={
    "roots.jsonl","worlds.jsonl","model_decisions.jsonl",
    "context_pairs_EVAL_ONLY.jsonl","witness_traces_EVAL_ONLY.jsonl",
    "answer_keys_EVAL_ONLY.jsonl","UPSTREAM_SOURCES.json","VALIDATION.json",
    "MANIFEST.sha256.json","CPU_BASELINES.json","HUMAN_REVIEW_PENDING.csv",
    "API_LANGUAGE_REVIEW_REQUIRED.jsonl","API_TEXT_RAW_CANDIDATES_REVIEW_REQUIRED.json",
    "API_LANGUAGE_USAGE.json","API_LANGUAGE_DIAGNOSTIC.json","README_CN.md",
}
assert all((DATA/f).is_file() for f in required)
manifest=json.loads((DATA/"MANIFEST.sha256.json").read_text(encoding="utf-8"))
for name,expected in manifest.items():
    assert sha256((DATA/name).read_bytes()).hexdigest()==expected,(name,"original hash mismatch")
source=[
    ROOT/"code"/"kci"/"knowledge_world_seed_v2.py",
    ROOT/"code"/"kci"/"public_workflow_ci.py",
    ROOT/"code"/"tests"/"test_knowledge_world_seed_v2.py",
    ROOT/"code"/"tests"/"test_public_workflow_ci.py",
    ROOT/"tools"/"build_public_workflow_ci.py",
    ROOT/"tools"/"run_public_ci_baselines.py",
    ROOT/"tools"/"draft_public_ci_language_once.py",
    ROOT/"tools"/"audit_public_ci_language.py",
]
licences={
    "THIRD_PARTY_LICENSES/AGENTDOJO_MIT_LICENSE.txt":
        Path(r"D:\research\kci_public_refs_20261119\agentdojo\LICENSE"),
    "THIRD_PARTY_LICENSES/TAU2_MIT_LICENSE.txt":
        Path(r"D:\research\kci_public_refs_20261119\tau2-bench\LICENSE"),
}
files={f"dataset/{x.name}":x for x in DATA.iterdir() if x.is_file()}
files.update({x.relative_to(ROOT).as_posix():x for x in source})
files.update(licences)
assert all(x.exists() for x in files.values())
# Include a complete bundle manifest and detailed checksums in a new entry,
# without modifying the original source data manifest.
entryhash={k:sha256(p.read_bytes()).hexdigest() for k,p in sorted(files.items())}
content_manifest=json.dumps(
    {"dataset":"KCI public workflow inspired CI 10-task executable pilot",
     "source":"synthetic, not official AgentDojo/tau2 benchmark task",
     "training_ready":False,"source_and_files_sha256":entryhash},
    indent=2,ensure_ascii=False,sort_keys=True
).encode("utf-8")
with ZipFile(TARGET,"x",ZIP_DEFLATED,compresslevel=8) as z:
    for name,src in sorted(files.items()):
        raw=src.read_bytes()
        # Check for accidental embedding of obvious secret key shapes.
        import re
        if re.search(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}",raw):
            raise RuntimeError("Possible API key in: "+name)
        z.writestr(name,raw)
    z.writestr("BUNDLE_SHA256_MANIFEST.json",content_manifest)
with ZipFile(TARGET,"r") as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    assert all(sha256(z.read(name)).hexdigest()==expected
               for name,expected in entryhash.items())
report={
    "status":"PACKAGE_VERIFIED",
    "file":str(TARGET),
    "bytes":TARGET.stat().st_size,
    "sha256":sha256(TARGET.read_bytes()).hexdigest(),
    "files":len(files)+1,
    "source_repositories":2,
    "task_roots":10,
    "worlds":70,
    "context_pairs":30,
    "cpu_tests_run_in_separate_step":True,
    "actual_4b_gpu_evaluation_completed":False,
    "API_key_included":False,
    "candidate_language_requires_human_review":True,
    "direct_train_ready":False
}
receipt=DATA/"PACKAGE_RECEIPT.json"
if receipt.exists():raise FileExistsError(receipt)
receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
