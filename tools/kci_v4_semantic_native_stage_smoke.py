"""Offline V4 dev full-team native action stage inspection; no 4B or labels."""
from pathlib import Path
from dataclasses import asdict
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.data_v4_env import V4ChainEnv, V31Case, Task, text_surface
D=ROOT/"artifacts/20261123_v4_semantic_control_v1"
def read(n):return [json.loads(x) for x in (D/n).read_text(encoding="utf8").splitlines()]
meta=read("DEV_ROOT_SPECS.jsonl")
worlds=read("DEV_WORLD_CONFIGS.jsonl")
r=meta[0]
sp=dict(r["native_v31_compatible_spec"]);sp["task"]=Task(**sp["task"])
env=V4ChainEnv(V31Case(**sp),worlds[0]["first_packet_kind"],worlds[0]["second_packet_kind"],
               worlds[0]["explicit_authorization"],worlds[0]["reversed_ledger_delivery_order"])
state=env.initial()
for i in range(30):
    if state.native.done:break
    obs=env.observation(state);p,choices=text_surface(obs)
    stage=obs.get("current_stage")
    print(i,"ACTOR",obs.get("actor"),"STAGE",stage,"ACTIONS",choices,flush=True)
    action=None
    if "A_READ_VERIFIED_PACKAGES" in choices:action="A_READ_VERIFIED_PACKAGES"
    elif worlds[0]["upstream_actual_action"] in choices:action=worlds[0]["upstream_actual_action"]
    elif "INSPECT_BUSINESS" in choices:action="INSPECT_BUSINESS"
    elif "CHOOSE:"+env.native._route in choices:action="CHOOSE:"+env.native._route
    elif "OPEN_PRIVATE_CHANNEL" in choices:action="OPEN_PRIVATE_CHANNEL"
    elif stage=="choose_fact":action=next(k for k in choices if k.startswith("READ:"))
    elif "VERIFY_PRIVATE_R" in choices:action="VERIFY_PRIVATE_R"
    elif "SEND_CERT_R" in choices:action="SEND_CERT_R"
    elif "COMMIT_RESULT" in choices:action="COMMIT_RESULT"
    else:break
    state,r=env.transition(state,action)
    print("  ",action,r["verdict"])
print("RESULT",state.native.done,state.native.success,state.native.violation,state.native.steps)
