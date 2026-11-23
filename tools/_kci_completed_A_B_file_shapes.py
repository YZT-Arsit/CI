from pathlib import Path
import json,collections
R=Path(r"D:\research\longhistory\artifacts")
A=R/"20261123_executor_coverage_completed_audit_v1"
B=R/"20261123_cce_reference_completion_v1"
for p in [A/"DECISION.json",A/"baseline_EVAL.json",A/"candidate_EVAL.json",A/"TRAINING.json",B/"ANALYSIS.json",B/"reference_COST.json",B/"STATE_POOL.json"]:
 d=json.loads(p.read_text(encoding="utf8"))
 print("\n===",p.name,str(p.parent),type(d).__name__,"===")
 print("TOP_KEYS",list(d)[:35] if isinstance(d,dict) else len(d))
 if isinstance(d,dict):
  for k in list(d)[:15]:
   v=d[k];print("field",k,repr(v)[:650])
