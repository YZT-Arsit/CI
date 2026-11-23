from pathlib import Path
import json,collections,hashlib
root=Path(r"D:\research\kci_ci_causal_coverage_v2_20261122")
folder=root/"artifacts/20261122_kci_causal_credit_method_handoff_v1"
for name in ["MODEL_ONLY_TRAIN_A.jsonl","MODEL_ONLY_TRAIN_DOWNSTREAM.jsonl",
             "TRAIN_EARLY_A_ACTION_CE_IDENTICAL_STATES.jsonl","TRAIN_ONE_PER_INDEPENDENT_STRICT_A_DPO.jsonl"]:
 rows=[json.loads(x) for x in (folder/name).open(encoding="utf8") if x.strip()]
 print("====",name,"ROWS",len(rows),"keys",list(rows[0]),"max_chars",max(len(x.get("model_prompt",x.get("prompt",""))) for x in rows),"min_chars",min(len(x.get("model_prompt",x.get("prompt",""))) for x in rows))
 print("actions",collections.Counter(tuple(r.get("available_actions",[])) for r in rows))
 print("stage",collections.Counter(r.get("stage") for r in rows))
 for r in rows[:2]:
  print("example",{"request_id":r.get("request_id"),"stage":r.get("stage"),"actor":r.get("actor"),"available_actions":r.get("available_actions"),"prompt_prefix":r.get("model_prompt","")[:350]},flush=True)
