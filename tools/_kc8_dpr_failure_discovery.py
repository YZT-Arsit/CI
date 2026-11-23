from pathlib import Path
import json
p=Path(r"D:\research\longhistory\artifacts\20261123_kc8_4b_policy_signal_v1\REAL4B_TRAIN_FROZEN_DIAGNOSTIC\REAL_4B_TRAIN_FULL_MODEL_NATIVE_EPISODES.jsonl")
rows=[json.loads(x) for x in p.open(encoding="utf8")]
print("n_rows",len(rows))
for r in rows:
    if r["CI_violation"]:
        a=next(x for x in r["real_model_action_trace"] if x["native_CI_violation_after"])
        print("FAILURE",r["root_id"],r["history"],r["seed"],"stage",a["native_stage"],"type",a["event_type"],"act",a["native_action_or_chance"],"idx",a["event_index"],"nprev",len(r["official_full_episode_record"]["events"]),flush=True)
for r in rows:
    if r["safe_abort"]:
        a=next(x for x in r["real_model_action_trace"] if x["native_action_or_chance"]=="ABORT")
        print("SAFE_ABORT",r["root_id"],r["history"],r["seed"],"stage",a["native_stage"],"idx",a["event_index"],flush=True)
