from pathlib import Path
import sys,json,collections
R=Path(r"D:\research\longhistory")
sys.path[:0]=[str(R/"tools"),str(R/"code")]
from kci_history_joint_static_visible_diagnostic import local_visible_policy
P=R/"artifacts/20261124_joint_history_supervision_v2/TRAIN_FIVE_CONTEXTS.jsonl"
rows=[json.loads(x) for x in P.open(encoding="utf8")]
c=collections.Counter();errs=[]
for x in rows:
 a,e=local_visible_policy(x["actor_visible_observation"])
 ok=a in x["acceptable_native_actions"]
 c[(x["condition"],ok)]+=1
 if not ok:errs.append((x["world"],a,x["acceptable_native_actions"]))
print("TRAIN_VISIBLE_STATIC_CLOSURE",sum(k[1]*n for k,n in c.items()),"/",len(rows),"by_condition",dict(c),"errors",errs)
