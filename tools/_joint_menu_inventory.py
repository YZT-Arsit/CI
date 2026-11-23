from pathlib import Path
import json,collections
p=Path(r"D:\research\longhistory\artifacts\20261124_joint_history_supervision_v1\TRAIN_FIVE_CONTEXTS.jsonl")
c=collections.defaultdict(set)
for l in p.open(encoding="utf8"):
 r=json.loads(l);c[r["root"]].add(tuple(r["available_native_actions"]))
for k,v in c.items():print(k,v)
