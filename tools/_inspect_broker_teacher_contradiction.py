from pathlib import Path
import json
p=Path(r"D:\research\kci_data_v4_phase2_20261122\artifacts\20261122_kci_data_v4_phase2_full_v1\FULL_NATIVE_ACTION_TRAJECTORIES_EVAL_ONLY.jsonl")
for line in p.open(encoding="utf8"):
 if '"P206' not in line and '"P207' not in line:continue
 r=json.loads(line)
 if r["root_id"] not in ("P206","P207"):continue
 print("\n",r["world_id"],flush=True)
 for action,x in r["native_action_branches"].items():
  print(action,"CI",x["immediate_verdict"],"succ",x["real_fixed_suffix"]["success"],"violate",x["real_fixed_suffix"]["violation"],"deadline",x["real_fixed_suffix"]["deadline_missed"],"steps",x["real_fixed_suffix"]["total_steps"],flush=True)
