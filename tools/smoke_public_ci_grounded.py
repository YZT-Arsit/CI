import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci_grounded import GroundedPublicCIEnv
from kci.public_workflow_ci import LOOKUP
from kci.knowledge_world_seed_v2 import find_safe_witness,topological_orders,FACTS
for name in ("W01","R03"):
 for h in ("linked","unrelated"):
  for profile in ("minimal","full"):
   env=GroundedPublicCIEnv(LOOKUP[name],h,False,profile)
   witness=find_safe_witness(env.knowledge)
   print("TEST",name,h,profile,witness["order"],witness["privately_verified"],flush=True)
   r=env.run_script(witness["order"],witness["privately_verified"])
   print("RESULT",name,h,profile,r["success"],r["violation"],len(r["tool_calls"]),flush=True)
