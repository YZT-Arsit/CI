"""Reconstruct all reached downstream states, retaining original generation batches."""
import hashlib
import json
import sys
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261117_fallback_order'
sys.path[:0]=[str(OLD),str(R.parent/'20261117_binding_factors')]
from worlds import targets,check
from experiments.relation_solution_data import environment
from kci.relation_language import seed_for
from render import actor,query
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=lambda p:list(map(json.loads,p.read_text(encoding='utf-8').splitlines()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def verify_prior():
    plan=read(OLD/'PLAN.json');seal=read(OLD/'SEALED.json')
    assert seal['complete']
    assert all(sha(OLD/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes'],**seal['files']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())


def construct_batches():
    pool=read(R.parent/'20261116_relation_solution/entity_palettes.json')['fresh'];result=[]
    first={(d['arm'],d['condition'],d['world_id']):d['first_nonoptimal'] for d in read(OLD/'FAILURE_LOCATIONS.json')['details']}
    for arm in ['initial','interleaved']:
        for condition,ws in read(OLD/'dev_worlds.json').items():
            records=rows(OLD/'repair'/arm/(condition+'.jsonl'));states={}
            assert len(records)==len(ws)
            for index,(w,old) in enumerate(zip(ws,records)):
                assert old['world_id']==w['world_id'];env=environment(w);state=env.initial()
                cases=json.loads(json.dumps(targets(w)))
                for entry in old['trace']:
                    obs=json.loads(json.dumps(env.observation(state)))
                    case=next(c for c in cases if c['observation']==obs);assert not check(case)
                    phase=entry['phase'];nonce=seed_for(w['base_group_id'],phase)
                    seeds={k:nonce for k in ['names','facts','plans','links','options']}
                    bits={q['action']:q['estimate'] for q in entry['queries']}
                    truth={c['action']:str(c['safe_completion']) for c in case['candidates']}
                    original,order=actor(obs,pool,seeds,bits);text,new_order=actor(obs,pool,seeds,truth)
                    assert original==entry['prompt'] and order==new_order==entry['actions']
                    assert all(q['prompt']==query(obs,pool,seeds,q['action']) for q in entry['queries'])
                    if phase>0:
                        failure=first[arm,condition,w['world_id']]
                        states[phase,index]={'arm':arm,'condition':condition,'index':index,'phase':phase,'world_id':w['world_id'],
                            'case':case,'prompt':text,'actions':order,'truth':truth,'original_estimates':bits,
                            'original_raw':entry['raw'],'original_selected':entry['selected'],'original_seconds':entry['seconds_batch'],
                            'identical_prompt':text==original,'first_nonoptimal_here':bool(failure and failure['phase']==phase)}
                    if entry['selected'] is not None:state,_=env.transition(state,entry['selected'])
            for phase in [1,2]:
                for start in range(0,len(ws),2):
                    batch=[states[phase,i] for i in range(start,min(start+2,len(ws))) if (phase,i) in states]
                    if batch:result.append(batch)
    return result


def main():
    assert not (R/'PROBE_PLAN.json').exists();verify_prior();batches=construct_batches()
    path=R/'probe_batches.jsonl';path.write_text(''.join(json.dumps(b)+'\n' for b in batches),encoding='utf-8')
    plan={'scope':'Fixed reached Source/Worker states, oracle estimates only; not full oracle rollouts or learned-query success',
        'arms':['initial','interleaved'],'responses':sum(map(len,batches)),'batches':len(batches),'model_updates':0,
        'prior_plan_sha256':sha(OLD/'PLAN.json'),'prior_seal_sha256':sha(OLD/'SEALED.json'),
        'prior_failure_locations_sha256':sha(OLD/'FAILURE_LOCATIONS.json'),'data_sha256':sha(path),
        'sources':{n:sha(R/n) for n in ['probe_data.py','probe.py','audit_probe.py']},
        'interpretation_gate':'Each arm has identical-prompt controls; zero raw disagreements on identical inputs; zero invalid oracle actions. Report first-error subset and all reached states separately.',
        'whole_goal_complete':False}
    (R/'PROBE_PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');print(json.dumps(plan))


if __name__=='__main__':main()
