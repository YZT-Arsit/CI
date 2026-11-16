"""Frozen finite policies under lawful prefix interventions; no training."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time
from experiments.text_policy_probe import action_class,forward,text_features,np
from experiments.multi_dataset import case_env
from kci.continuation import ContinuationEvaluator
from kci.multi_candidates import MultiRuleCandidateGenerator
from kci.contracts import Verdict


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--raw',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();start=time.perf_counter()
    model=Path(a.model);out=Path(a.output)
    if out.exists():raise FileExistsError('new evaluation output required')
    weights=np.load(model/'weights.npz');u=weights['u'];e=weights['e']
    result=json.loads((model/'result.json').read_text(encoding='utf-8'))
    generator=MultiRuleCandidateGenerator();rows=[];inputs={}
    for path in sorted(Path(a.raw).glob('*.json')):
        case=json.loads(path.read_text(encoding='utf-8'))
        if case['case_spec']['family']!=result['heldout_family']:continue
        inputs[path.name]=sha256(path.read_bytes()).hexdigest();env=case_env(case['case_spec'])
        values={s['state_id']:s for s in case['states']}
        for intervention in ('skip','context'):
            initial=env.initial()
            first=env.act('A','SKIP') if intervention=='skip' else env.act('A','SEND','R',env.context_ids)
            state,audit=env.transition(initial,first);assert audit.verdict==Verdict.VALID
            state_row=values[ContinuationEvaluator.state_id(state)]
            costs=[x['minimum_safe_total_cost'] for x in state_row['actions'] if x['minimum_safe_total_cost'] is not None]
            optimal=min(costs) if costs else None
            actions=[];status='TASK_FAILURE'
            while not state.done:
                obs=env.observation(state,state.actor);menu=generator.generate(obs,12).actions
                _,scores=forward(text_features(obs.as_prompt())[None],u,e)
                ids=[action_class(act.__dict__) for act in menu]
                action=menu[int(np.argmax(scores[0,ids]))]
                state,check=env.transition(state,action)
                actions.append({'action':action.as_json(),'verdict':check.verdict.value})
                if check.verdict!=Verdict.VALID:
                    status='VIOLATION' if check.verdict==Verdict.VIOLATION else 'UNKNOWN';break
            else:status='SAFE_SUCCESS' if state.success and not state.any_violation else 'TASK_FAILURE'
            rows.append({'case_id':case['case_spec']['case_id'],'intervention':intervention,
                         'prefix_action':first.as_json(),'feasible':optimal is not None,'optimal_total_cost':optimal,
                         'status':status,'final_cost':state.cost,'actions':actions})
    summary={'heldout_family':result['heldout_family'],'variant':result['variant'],'seed':result['seed'],
             'episodes':len(rows),'counts':dict(Counter(r['status'] for r in rows)),
             'by_intervention':{s:dict(Counter(r['status'] for r in rows if r['intervention']==s)) for s in ('skip','context')},
             'feasible_counts':dict(Counter(r['status'] for r in rows if r['feasible'])),
             'infeasible_counts':dict(Counter(r['status'] for r in rows if not r['feasible'])),
             'rows':rows,'model_weights_sha256':sha256((model/'weights.npz').read_bytes()).hexdigest(),
             'input_sha256':inputs,'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
             'seconds':time.perf_counter()-start,
             'scope':'exploratory lawful-prefix intervention on frozen finite models; NOT new training or real LLM evaluation'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('episodes','counts','feasible_counts','seconds')}))


if __name__=='__main__':main()
