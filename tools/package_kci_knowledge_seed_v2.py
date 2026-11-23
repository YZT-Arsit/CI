"""Produce a self-contained KCI receiver-knowledge seed pilot zip without secrets."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile,ZIP_DEFLATED
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"artifacts"/"20261119_composition_signals"/"knowledge_seed_v2"
ZIP=ROOT/"artifacts"/"20261119_composition_signals"/"KCI_knowledge_seed_v2_pilot.zip"
if ZIP.exists():raise FileExistsError(ZIP)
raw=[json.loads(s) for s in (DATA/"roots.jsonl").read_text(encoding="utf-8").splitlines()]
keys={x["seed_id"]:x for x in (json.loads(s) for s in (DATA/"answer_key_EVAL_ONLY.jsonl").read_text(encoding="utf-8").splitlines())}
assert len(raw)==len(keys)==20
sheet=DATA/"HUMAN_REVIEW_SHEET.csv"
if sheet.exists():raise FileExistsError(sheet)
with sheet.open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=[
        "seed_id","domain","rule_topology","workflow_mechanism","task_description",
        "sensitive_relation","history_change_validated","order_early_step","order_late_step",
        "external_norm_verified","manual_task_review","manual_privacy_norm_review",
        "manual_business_source_review","reviewer_notes"])
    writer.writeheader()
    for x in raw:
        k=keys[x["seed_id"]]
        writer.writerow({
            "seed_id":x["seed_id"],"domain":x["domain"],"rule_topology":x["topology"],
            "workflow_mechanism":x["mechanism_signature"],
            "task_description":x["task_goal"],
            "sensitive_relation":x["protected_relation"],
            "history_change_validated":bool(k["flip_true"]),
            "order_early_step":k["order_early_violation_step"],
            "order_late_step":k["order_late_violation_step"],
            "external_norm_verified":"NO",
            "manual_task_review":"PENDING","manual_privacy_norm_review":"PENDING",
            "manual_business_source_review":"PENDING",
            "reviewer_notes":"",
        })
attempt=DATA/"API_TEXT_ATTEMPT_STATUS.json"
if attempt.exists():raise FileExistsError(attempt)
attempt.write_text(json.dumps({
    "requested_purpose":"fictional prose paraphrases only",
    "api_key_configured_without_exposure":True,
    "one_response_parse_completed":True,
    "response_lost_before_disk_write_due_to_client_script_key_name_bug":True,
    "language_candidates_saved":False,
    "retry_submitted":False,
    "response_store":False,
    "usage_tokens":"unknown_not_retained",
    "backend_structured_dataset_unchanged":True,
    "human_review_still_required":True,
},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
src=[
    ROOT/"code"/"kci"/"knowledge_world_seed_v2.py",
    ROOT/"code"/"tests"/"test_knowledge_world_seed_v2.py",
    ROOT/"tools"/"build_kci_knowledge_seed_v2.py",
    ROOT/"tools"/"audit_kci_knowledge_seed_v2.py",
    ROOT/"tools"/"enrich_kci_seed_language_once.py",
]
assert all(x.is_file() for x in src)
datasets=sorted(x for x in DATA.iterdir() if x.is_file())
assert len(datasets)>=12
with ZipFile(ZIP,"w",compression=ZIP_DEFLATED,compresslevel=7) as z:
    for f in sorted(datasets+src):
        name=f.relative_to(ROOT).as_posix()
        if any(part in ("cache",".venv",".git","__pycache__") for part in f.parts):
            raise RuntimeError("unexpected private source")
        content=f.read_bytes()
        assert b"sk-proj-" not in content and b"sk-test-" not in content
        z.writestr(name,content)
with ZipFile(ZIP) as z:
    assert z.testzip() is None
    contains=z.namelist()
report={"package_path":str(ZIP),"bytes":ZIP.stat().st_size,
        "SHA256":sha256(ZIP.read_bytes()).hexdigest(),
        "files":len(contains),"status":"COMPLETE",
        "no_secrets_bundled":True,"API_language_candidates":False,
        "dataset_training_ready":False}
receipt=DATA/"PACKAGE_RECEIPT.json"
receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
