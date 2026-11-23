from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci_remaining_grounded import RemainingGroundedCIEnv
from kci.public_workflow_ci import LOOKUP
from kci.knowledge_world_seed_v2 import find_safe_witness
for rid in ("W04","W05"):
 env=RemainingGroundedCIEnv(LOOKUP[rid],"linked",False,"full")
 witness=find_safe_witness(env.knowledge)
 st=env.initial()
 for f in witness["order"]:
  if st.native.done:break
  for action in (env.encode_action("PROCESS_"+f),
          env.encode_action("VERIFY_PRIVATE_"+f) if f in witness["privately_verified"]
          else env.encode_action("SEND_RAW_"+f)):
   st,a=env.transition(st,action)
   print(rid,action,"->",a,"native",st.native.phase,st.native.turn,flush=True)
   if a["verdict"]!="VALID":break
  if f in witness["privately_verified"] and st.native.phase=="attest":
   act=env.encode_action("SEND_ATTEST_"+f)
   st,a=env.transition(st,act)
   print(rid,act,"->",a,"native",st.native.phase,st.native.turn,flush=True)
 print("END",rid,"options",env.options(st),"state",st.native.phase,st.native.turn,flush=True)
