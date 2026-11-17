"""Label-only contextualization of DEV success denominators; no model outputs read."""
import hashlib
import json
import sys
from pathlib import Path

R=Path(__file__).resolve().parent
sys.path[:0]=[str(R.parents[1]/'code'),str(R.parent/'20261116_relation_solution')]
from experiments.relation_solution_data import targets
from audit_data import check

worlds_path=R/'dev_worlds.json'
plan=json.loads((R/'PLAN.json').read_text())
assert hashlib.sha256(worlds_path.read_bytes()).hexdigest()==plan['dev_sha256']
worlds=json.loads(worlds_path.read_text()); records=[]
for condition, group in worlds.items():
    for world in group:
        initial=targets(world)[0]
        assert initial['observation']['role']=='Planner' and not check(initial)
        records.append({'condition':condition,'world_id':world['world_id'],
                        'safe_completion_exists':any(c['safe_completion'] for c in initial['candidates'])})
result={'world_presentations':len(records),'feasible_presentations':sum(r['safe_completion_exists'] for r in records),
        'infeasible_presentations':sum(not r['safe_completion_exists'] for r in records),
        'records':records,'model_outputs_read':False,'model_calls':0,
        'registration':'Added during inference, before full result scoring; contextualizes original denominators without replacing preregistered primary metrics.',
        'scope':'Finite native synthetic task feasibility; not unrestricted real-world impossibility.'}
(R/'TASK_FEASIBILITY.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
