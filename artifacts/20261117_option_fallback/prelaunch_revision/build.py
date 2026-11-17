"""Freeze all six option orders, with original semantic observations unchanged."""
import hashlib
import itertools
import json
import sys
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path[:0]=[str(ROOT/'code'),str(R.parent/'20261117_binding_factors')]
from render import actor
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'PLAN.json').exists()
old=R.parent/'20261117_frozen_crossover'
train=list(map(json.loads,(old/'train_oracle.jsonl').read_text().splitlines()))
worker=[row for row in train if row['observation']['role']=='Worker']
zero=[row for row in worker if not any(c['safe_completion'] for c in row['candidates'])]
positive=sorted([row for row in worker if any(c['safe_completion'] for c in row['candidates'])],key=lambda r:hashlib.sha256(r['case_id'].encode()).hexdigest())[:18]
assert len(zero)==len(positive)==18
pool=json.loads((R.parent/'20261116_relation_solution/entity_palettes.json').read_text())['train']
observations=[];labels=[]
for group,cases in [('zero',zero),('positive',positive)]:
    for case in cases:
        obs=case['observation'];bits={c['action']:str(c['safe_completion']) for c in case['candidates']}
        original,original_order=actor(obs,pool,case['seeds'],bits)
        available={}
        for seed in range(1000):
            seeds={**case['seeds'],'options':seed}
            text,order=actor(obs,pool,seeds,bits)
            available.setdefault(tuple(order),(seed,text))
            if len(available)==6:break
        assert set(available)==set(itertools.permutations(obs['actions']))
        for variant,order in enumerate(sorted(available)):
            seed,text=available[order];identifier=f"{case['case_id']}_order{variant}"
            observations.append({'id':identifier,'case_id':case['case_id'],'group':group,'prompt':text,
                'actions':list(order),'options_seed':seed,'original_prompt':text==original})
            labels.append({'id':identifier,'optimal_actions':[case['candidates'][i]['action'] for i in case['optimal_indices']],
                           'candidates':case['candidates']})
for name,values in [('observations',observations),('labels',labels)]:
    (R/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in values))
checkpoint=R.parent/'20261117_optimization_repair/repair/continue_lower_lr'
prior=checkpoint/'train_fit.jsonl'
plan={'checkpoint':checkpoint.relative_to(ROOT).as_posix(),'checkpoint_seal':json.loads((checkpoint/'CHECKPOINT_SEALED.json').read_text()),
      'cases':36,'presentations':216,'oracle_estimates':True,'model_updates':0,
      'data_hashes':{n:sha(R/n) for n in ['observations.jsonl','labels.jsonl']},
      'dependencies':{'artifacts/20261117_frozen_crossover/train_oracle.jsonl':sha(old/'train_oracle.jsonl'),
                      prior.relative_to(ROOT).as_posix():sha(prior)},
      'sources':{p.name:sha(p) for p in R.glob('*.py')},'whole_goal_complete':False}
(R/'PLAN.json').write_text(json.dumps(plan,indent=2))
print(json.dumps({'cases':36,'presentations':216}))
