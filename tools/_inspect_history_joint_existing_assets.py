from collections import Counter,defaultdict
from pathlib import Path
import json
p=Path(r"D:\research\kci_data_v4_phase2_20261122\artifacts\20261122_kci_data_v4_phase2_full_v1")
files=["MODEL_ROLE_OBSERVATIONS_EVAL_ONLY.jsonl","CONTROLLED_CONTEXT_PAIRS_EVAL_ONLY.jsonl","FULL_NATIVE_ACTION_TRAJECTORIES_EVAL_ONLY.jsonl","TRAIN_SAME_STATE_PREFERENCES.jsonl","NEW_WORLDS.jsonl"]
for f in files:
 xs=[]
 for line in (p/f).open(encoding="utf8"):
  if '"final_test"' in line:continue
  x=json.loads(line)
  if x.get("split") in ("train","dev") or x.get("root_id","").startswith("P2") and x.get("root_id") not in ("P211","P212"):xs.append(x)
 print("===== ",f,len(xs),"=====",flush=True)
 print("keys",list(xs[0]),flush=True)
 print("roots",Counter(x.get("root_id") for x in xs),flush=True)
 print("splits",Counter(x.get("split") for x in xs),flush=True)
 for x in xs[:3]:
  print("row:",{k:repr(v)[:900] for k,v in x.items() if k in (
  "root_id","world_id","split","variant","kind","category","pair_id","left_world","right_world","same_current_signed_source","same_history_size",
  "role_observation_sha256","native_state_sha256","true_gold","action","chosen_action","rejected_action","native_action_branches","root_split","source_truth")},flush=True)
 if f.startswith("MODEL"):
  for x in xs[:3]:
   print("model fields",(x.get("world_id"),x.get("root_id"),x.get("role_observation",{}).get("stage"),x.get("role_observation",{}).get("legal_actions")),flush=True)
