"""GPU transfer bundle: KCI state-directed restart baseline plus TRAIN fork set.
No secrets, historical model weights, original experiments or cached tensors.
"""
from hashlib import sha256
from pathlib import Path
import json,zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
assert ROOT.name=="longhistory"
paths=list((ROOT/"code"/"kci").rglob("*.py"))
paths += [ROOT/"code"/"experiments"/"__init__.py",
          ROOT/"code"/"experiments"/"receipt_dataset.py",
          ROOT/"tools"/"run_kci_state_directed.py",
          ROOT/"tools"/"audit_kci_state_restart.py",
          ROOT/"tools"/"audit_kci_policy_qpi.py",
          ROOT/"tools"/"audit_kci_agent_rl.py",
          ROOT/"tools"/"run_kci_agent_rl.py",
    HERE/"STATE_DIRECTED_MATCHED_PLAN.md",
          HERE/"selection_v2"/"FORKS.jsonl",
          HERE/"selection_v2"/"SELECTION.json"]
paths += [p for p in (ROOT/"artifacts"/"20261113_iteration2"/"source_anchors").rglob("*")
          if p.is_file()]
indexed={p.relative_to(ROOT).as_posix():p for p in paths}
assert len(indexed)==len(paths)
target=HERE/"gpu_state_restart_matched_v2.zip"
if target.exists():raise FileExistsError(target)
with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(indexed.items()):z.write(p,name)
obj={"SHA256":sha256(target.read_bytes()).hexdigest(),
     "bytes":target.stat().st_size,"files":len(indexed),
     "source_hashes":{name:sha256(p.read_bytes()).hexdigest() for name,p in sorted(indexed.items())},
     "contains_private_credentials":False,"contains_model_checkpoints":False,
     "selection_train_only_not_blind_test":True,
     "gpu_training_not_auto_enabled":True}
(HERE/"STATE_RESTART_MATCHED_V2_MANIFEST.json").write_text(
    json.dumps(obj,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in obj.items() if k!="source_hashes"}))
