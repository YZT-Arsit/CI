"""Post-result descriptive localization on realized trajectories, not causal mediation."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from worlds import targets
from experiments.relation_solution_data import environment

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(R/'PLAN.json');seal=read(R/'SEALED.json')
assert seal['complete'] and all(sha(R/n)==h for n,h in seal['files'].items())
assert all(sha(R/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes']}.items())
assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
summaries=[];details=[]
for arm in plan['arms']:
    stats=Counter();errors=defaultdict(Counter)
    for condition,worlds in read(R/'dev_worlds.json').items():
        recorded=rows(R/'repair'/arm/(condition+'.jsonl'))
        for w,row in zip(worlds,recorded):
            assert w['world_id']==row['world_id'];cases=json.loads(json.dumps(targets(w)));env=environment(w);state=env.initial()
            initial_feasible=any(c['safe_completion'] for c in cases[0]['candidates']);first=None
            for phase,entry in enumerate(row['trace']):
                obs=json.loads(json.dumps(env.observation(state)));case=next(c for c in cases if c['observation']==obs)
                best={case['candidates'][i]['action'] for i in case['optimal_indices']};chosen=entry['selected']
                truth={c['action']:str(c['safe_completion']) for c in case['candidates']};pred={q['action']:q['estimate'] for q in entry['queries']}
                if first is None and chosen not in best:
                    positive_best={a for a in best if truth[a]=='1'}
                    first={'phase':phase,'role':obs['role'],'selected':chosen,'optimal_actions':sorted(best),'truth':truth,'predicted':pred,
                        'estimates_exact':pred==truth,'all_positive_optima_predicted_zero':bool(positive_best) and all(pred[a]=='0' for a in positive_best),
                        'selected_false_positive':chosen is not None and pred[chosen]=='1' and truth[chosen]=='0'}
                if chosen is not None:state,_=env.transition(state,chosen)
            stats.update(episodes=1,initial_feasible=int(initial_feasible),strict_success=int(row['strict_success']),violations=int(not row['ci']))
            if first:
                errors[first['role']].update(first_nonoptimal=1,estimates_exact=int(first['estimates_exact']),
                    positive_optimum_predicted_zero=int(first['all_positive_optima_predicted_zero']),selected_false_positive=int(first['selected_false_positive']))
            else:
                stats.update(no_nonoptimal_action=1,no_nonoptimal_infeasible=int(not initial_feasible),
                             no_nonoptimal_but_strict_failure=int(not row['strict_success']))
            details.append({'arm':arm,'condition':condition,'world_id':w['world_id'],'initial_feasible':initial_feasible,'first_nonoptimal':first})
    summaries.append({'arm':arm,**dict(stats),'by_role':{role:dict(c) for role,c in errors.items()}})
(R/'FAILURE_LOCATIONS.json').write_text(json.dumps({'scope':'Realized-trajectory first nonoptimal actions. Trajectories differ by arm; estimates/error co-occurrence is descriptive, not causal attribution.',
    'summaries':summaries,'details':details,'model_calls':0,'primary_gate_changed':False},indent=2))
print(json.dumps(summaries))
