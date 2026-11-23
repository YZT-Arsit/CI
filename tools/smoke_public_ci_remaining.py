from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci_remaining_grounded import RemainingGroundedCIEnv,REMAINING
from kci.public_workflow_ci import LOOKUP
from kci.knowledge_world_seed_v2 import find_safe_witness,topological_orders,FACTS
for rid in REMAINING:
 print("TASK",rid,flush=True)
 for h in ("linked","unrelated"):
  for profile in ("minimal","full"):
   env=RemainingGroundedCIEnv(LOOKUP[rid],h,False,profile)
   witness=find_safe_witness(env.knowledge)
   assert witness,("no finite native safe completion",rid,h,profile)
   x=env.run_script(witness["order"],witness["privately_verified"])
   print(rid,h,profile,
         "success",x["success"],"violation",x["violation"],
         "done",x["done"],"calls",len(x["tool_calls"]),
         flush=True)
