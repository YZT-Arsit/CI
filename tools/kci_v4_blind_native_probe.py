"""Read-only probe of actual V4 history-blind native teacher on TRAIN roots only."""
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
import importlib.util,json,sys
MAIN=Path(__file__).resolve().parents[1]
OTHER=Path(r"D:\research\kci_data_v4_pilot_20261122")
sys.path[:0]=[str(MAIN/"code"),str(MAIN/"tools")]
from kci.diverse_ci_branch_env import Task,digest
from kci.diverse_ci_long_horizon_v31 import V31Case
V4_MOD=OTHER/"code/kci/data_v4_env.py"
spec=importlib.util.spec_from_file_location("kci.data_v4_env",V4_MOD)
mod=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)
base=OTHER/"artifacts/20261122_kci_data_v4_pilot_v1"
def trainrows(n):
    for line in (base/n).open(encoding="utf8"):
        if '"DG36' in line or '"final_test"' in line:continue
        r=json.loads(line)
        if r.get("split")=="train":yield r
roots={}
for r in trainrows("ROOTS.jsonl"):
    sp=dict(r["native_v31_compatible_spec"])
    sp["task"]=Task(**sp["task"])
    roots[r["root_id"]]=V31Case(**sp)
assert sorted(roots)==["DG31","DG32","DG33"]
worlds=list(trainrows("WORLD_CONFIGS.jsonl"))
assert len(worlds)==15
result=[]
for w in worlds:
    if w["kind"] not in ("risky","safe","authorized"):continue
    env=mod.V4ChainEnv(roots[w["root_id"]],w["first_packet_kind"],w["second_packet_kind"],
                       w["explicit_authorization"],w["reversed_ledger_delivery_order"])
    state,_=mod.native_before_release(env,w["upstream_actual_action"].removeprefix("A_SEND_"))
    assert digest(asdict(state))==w["native_state_hash"],(w["root_id"],w["kind"],"hash mismatch")
    obs=env.observation(state)
    assert digest(obs)==w["observation_hash"]
    clone=deepcopy(state)
    clone.native.delivered["R"]=[]
    clone.native.ledger=[e for e in clone.native.ledger if e["recipient"]!="R"]
    raw=mod.native_continue(env,clone,"SEND_RAW_R")
    verified=mod.native_continue(env,clone,"VERIFY_PRIVATE_R")
    next_,v=env.transition(clone,"SEND_RAW_R")
    assert v["verdict"]=="VALID",w["world_id"]
    assert raw["terminal_success"] and verified["terminal_success"],w["world_id"]
    assert raw["execution_steps"]<verified["execution_steps"]
    result.append({"root":w["root_id"],"condition":w["kind"],"native_state_hash":w["native_state_hash"],
                   "blind_teacher_raw":"SEND_RAW_R","blind_raw_result":"VALID",
                   "blind_raw_steps":raw["execution_steps"],"blind_verify_steps":verified["execution_steps"],
                   "blind_teacher_no_knowledge_R_receipts":True})
assert len(result)==9
print(json.dumps({"result":"V4_NATIVE_BLIND_TRAIN_VALIDATED","n":len(result),
                  "root_counts":{k:sum(v["root"]==k for v in result) for k in roots},
                  "chosen_RAW":len(result),"individual":result},ensure_ascii=False,indent=2))
