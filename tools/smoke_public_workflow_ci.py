from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"code"))
from kci.public_workflow_ci import PUBLIC_TASKS,PublicCIEnv,check_goal
from kci.knowledge_world_seed_v2 import find_safe_witness,topological_orders,FACTS
for spec in PUBLIC_TASKS:
 env=PublicCIEnv(spec,"linked",False,"full")
 safe=find_safe_witness(env.knowledge)
 assert safe is not None
 try:
  r=env.run_script(safe["order"],safe["privately_verified"])
  base=env.run_script(topological_orders(env.seed,FACTS)[0],())
  inverse=PublicCIEnv(spec,"unrelated",False,"full")
  control=inverse.run_script(safe["order"],safe["privately_verified"])
  print(spec.id,"safe=",r["success"],"violation=",r["violation"],"raw_leak=",base["violation"],"tool_calls=",len(r["tool_calls"]),"unrelated_safe=",control["success"],"steps=",r["native_steps"])
 except Exception as e:
  print(spec.id,type(e).__name__,str(e))
  raise
