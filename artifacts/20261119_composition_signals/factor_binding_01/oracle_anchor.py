"""Frozen oracle-factor reference versus self-factor, same archived start only.

Old truth prompt is externally supplied. This is a DIAGNOSTIC comparator,
not autonomous relation learning and not a new training result.
"""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
old=ROOT/"artifacts"/"20261117_composition_repair"/"RESULTS.json"
current=HERE/"study"/"initial"/"NATIVE_RESULTS.json"
x=json.loads(old.read_text(encoding="utf-8"))
y=json.loads(current.read_text(encoding="utf-8"))
d=x["arms"]["truth17"]["fresh"]["modes"]
fields=("query_correct","actions_optimal","unsafe_actions")
source={k:{f:d[k][f] for f in fields} for k in ("direct","truth")}
cur={}
for mode,key in (("direct","direct"),("self_fact","self_generated_factor")):
    v=y[key]["summary"] if key=="direct" else y[key]["totals"]
    cur[mode]={"query_correct":v["query_correct"],"actions_optimal":v["action_optimal"],
               "unsafe_actions":v["immediate_unsafe"]}
assert cur["direct"]==source["direct"],"Old archived run not comparable to new start"
ref={"scope":"same truth17 archived starting model / 80 frozen states / no new model call",
     "archived_source_sha256":hashlib.sha256(old.read_bytes()).hexdigest(),
     "archived_oracle_external_truth":source["truth"],
     "archived_original_direct":source["direct"],
     "current_start_self_generated_relation":cur["self_fact"],
     "current_start_direct":cur["direct"],
     "interpretation":"True relation note is an external diagnosis and can help; noisy self-produced relation can harm. Does not prove a causal internal mediator."}
(HERE/"ORACLE_REFERENCE.json").write_text(json.dumps(ref,indent=2),encoding="utf-8")
print(json.dumps(ref))
