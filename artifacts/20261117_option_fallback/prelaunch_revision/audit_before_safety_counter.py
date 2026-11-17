"""Rebuild exact option interventions and retain tied-action correctness."""
import hashlib
import json
import re
import sys
from collections import Counter,defaultdict
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path[:0]=[str(ROOT/'code'),str(R.parent/'20261117_binding_factors')]
from render import actor
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))


def main():
    plan=read(R/'PLAN.json');run=read(R/'RUN_PLAN.json');seal=read(R/'SEALED.json')
    assert run['plan_sha256']==sha(R/'PLAN.json') and run['oracle_estimates'] and not run['reads_action_labels']
    assert all(sha(R/n)==h for n,h in {**plan['sources'],**plan['data_hashes']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
    assert seal['complete'] and all(sha(R/n)==h for n,h in seal['files'].items())
    checkpoint=ROOT/plan['checkpoint'];assert read(checkpoint/'CHECKPOINT_SEALED.json')==plan['checkpoint_seal']
    assert all(sha(checkpoint/'adapter'/p)==h for p,h in plan['checkpoint_seal']['adapter_hashes'].items())
    original={x['case_id']:x for x in rows(R.parent/'20261117_frozen_crossover/train_oracle.jsonl')}
    prior={x['case_id']:x['actor'] for x in rows(checkpoint/'train_fit.jsonl')}
    pool=read(R.parent/'20261116_relation_solution/entity_palettes.json')['train']
    observations=rows(R/'observations.jsonl');labels=rows(R/'labels.jsonl');responses=rows(R/'responses.jsonl')
    assert len(observations)==len(labels)==len(responses)==216
    stats=defaultdict(Counter);letters=defaultdict(Counter);cases=defaultdict(list);repeat=Counter();timings=[]
    for i,(obs,label,row) in enumerate(zip(observations,labels,responses)):
        assert obs['id']==label['id']==row['id']
        case=original[obs['case_id']];bits={c['action']:str(c['safe_completion']) for c in case['candidates']}
        text,order=actor(case['observation'],pool,{**case['seeds'],'options':obs['options_seed']},bits)
        assert text==obs['prompt'] and order==obs['actions'] and label['candidates']==case['candidates']
        best={case['candidates'][j]['action'] for j in case['optimal_indices']};assert set(label['optimal_actions'])==best
        found=re.findall(r'ANSWER:\s*([ABC])\b',row['raw']);valid=len(found)==1 and row['seconds_batch']<9.5
        selected=order[ord(found[0])-65] if valid else None
        key=(obs['group'],order.index('PUBLISH_FULL'))
        stats[key].update(presentations=1,optimal=int(selected in best),full=int(selected=='PUBLISH_FULL'),invalid=int(not valid))
        letters[obs['group']][found[0] if valid else 'invalid']+=1
        cases[obs['case_id']].append({'order':order,'optimal':selected in best,'group':obs['group']})
        assert obs['original_prompt']==(text==prior[obs['case_id']]['prompt'])
        if obs['original_prompt']:
            repeat.update(prompts=1,raw_disagreements=int(row['raw']!=prior[obs['case_id']]['raw']))
        if i%4==0:
            assert len({r['seconds_batch'] for r in responses[i:i+4]})==1
            timings.append(row['seconds_batch'])
    assert len(cases)==36 and repeat['prompts']==36
    worker=[r for r in original.values() if r['observation']['role']=='Worker']
    expected_zero={r['case_id'] for r in worker if not any(c['safe_completion'] for c in r['candidates'])}
    expected_positive={r['case_id'] for r in sorted([r for r in worker if any(c['safe_completion'] for c in r['candidates'])],
                      key=lambda r:hashlib.sha256(r['case_id'].encode()).hexdigest())[:18]}
    assert {cid for cid,vs in cases.items() if vs[0]['group']=='zero'}==expected_zero
    assert {cid for cid,vs in cases.items() if vs[0]['group']=='positive'}==expected_positive
    for values in cases.values():assert len(values)==len({tuple(v['order']) for v in values})==6
    result={'positions':[{'group':g,'full_position':'ABC'[p],**dict(v)} for (g,p),v in stats.items()],
            'letters':{g:dict(v) for g,v in letters.items()},
            'all_six_optimal':{g:sum(all(v['optimal'] for v in values) for values in cases.values() if values[0]['group']==g) for g in ['zero','positive']},
            'identical_prompt_control':dict(repeat),'responses':216,'batches':len(timings),
            'max_seconds':max(timings),'seconds_sum':sum(timings),'errors':0,'oracle_estimates':True,'whole_goal_complete':False}
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
