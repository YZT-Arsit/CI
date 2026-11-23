from pathlib import Path
import sys,json
R=Path(r"D:\research\longhistory")
sys.path[:0]=[str(R/"tools"),str(R/"code")]
from kci_history_joint_freeze_v2 import native_by_world,accept,V1
src=native_by_world()
for split in ("TRAIN","DEV"):
 for line in (V1/(split+"_FIVE_CONTEXTS.jsonl")).open(encoding="utf8"):
  r=json.loads(line);a,b,detail,reason=accept(src[r["world"]]["native_action_branches"],r["available_native_actions"])
  print(split,r["world"],"accept",a,"reject",b,"reason",reason,"info",[(act,detail[act]["immediate"],detail[act]["native_valid_complete_success"],detail[act]["native_steps"]) for act in detail])
