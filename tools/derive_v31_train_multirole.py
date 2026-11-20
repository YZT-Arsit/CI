r"""Create strictly train-graph-only multi-agent native optimal demonstrations.

The available future completion traces are oracle-generated. These records
can support supervised warmup/protocol testing, but must never be represented
as Qwen on-policy trajectories, Qpi labels, or real business data.
"""
from pathlib import Path
from collections import Counter
import json,sys
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"20261120_v31"/"V31_GATEA_FINAL_CANDIDATE"
SRC=OUT/"MULTI_ROLE_ORACLE_ACTION_OBSERVATIONS_EVAL_ONLY.jsonl"
TARGET=OUT/"TRAIN_ONLY_NATIVE_MULTIROLE_ACTION_DEMOS.jsonl"
REPORT=OUT/"TRAIN_MULTIROLE_AUDIT.json"
if TARGET.exists() or REPORT.exists():raise FileExistsError("Already derived")
rows=[json.loads(s) for s in SRC.read_text(encoding="utf-8").splitlines() if s.strip()]
selected=[]
duplicates=set()
counts=Counter()
for x in rows:
    if x["split"]!="train":continue
    if not x["world_id"].endswith(("__related","__same_entity_safe")):continue
    inp=x["model_visible_observation"]
    key=(x["world_id"],x["native_before_sha256"])
    if key in duplicates:continue
    duplicates.add(key)
    assert x["chosen_native_oracle_action"] in inp["legal_actions"]
    assert "expected_action" not in inp and "Qstar" not in inp
    selected.append({
        "id":x["world_id"]+"__"+str(x["current_step"]),
        "source_graph":x["graph_family"],"split":"train",
        "world_id":x["world_id"],"agent_role":x["role"],
        "decision_stage":x["stage"],
        "prompt":inp,
        "prompt_sha256":x["model_observation_sha256"],
        "oracle_native_action":x["chosen_native_oracle_action"],
        "source_native_state_sha256":x["native_before_sha256"],
        "provenance":"EXACT_NATIVE_ORACLE_DEMONSTRATION_NOT_LLM_ONPOLICY",
        "eligible_for_Qpi_preference_learning":False,
        "no_human_norm_certification":True,
    })
    counts[x["role"]]+=1
assert len(selected)>200 and len(counts)>=4,(len(selected),dict(counts))
with TARGET.open("x",encoding="utf-8") as file:
    for row in selected:file.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
audit={
    "status":"TRAIN_ONLY_NATIVE_MULTIROLE_REFERENCE_READY",
    "samples":len(selected),
    "distinct_roles":dict(counts),
    "training_graphs_only":True,
    "source_kind":"native optimal scripted continuations",
    "model_onpolicy_samples":0,
    "Qpi_training_labels":0,
    "notes":"Not a complete realistic multi-agent learned policy training corpus; stage distribution follows exact simulator optimal paths.",
    "sha256":sha256(TARGET.read_bytes()).hexdigest(),
}
REPORT.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
