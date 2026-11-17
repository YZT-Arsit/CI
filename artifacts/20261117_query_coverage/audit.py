"""Native evidence audit; fixed-state and autonomous metrics remain separate."""
import collections
import itertools
import json
import math
import re
import sys
from probe_data import R,ROOT,OLD,read,rows,sha,verify_prior
from training_data import schedule
from eval_worlds import targets,check,dev_worlds
from render import actor,query
from kci.relation_language import seed_for
sys.path.insert(0,str(R.parent/'20261117_teacher_binding'))
import replay


def main():
    plan=read(R/'PLAN.json');seal=read(R/'SEALED.json');verify_prior();folder=R/'repair'
    assert seal['complete'] and all(sha(R/n)==h for n,h in seal['files'].items())
    assert all(sha(R/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
    assert sha(R/'PROBE_SEALED.json')==plan['probe_seal_sha256']
    assert read(folder/'PLAN.json')==plan and read(folder/'dev_worlds.json')==read(R/'dev_worlds.json')
    assert read(R/'dev_worlds.json')==dev_worlds()[0]
    batches=rows(R/'train_batches.jsonl');assert batches==schedule()
    assert len(batches)==96 and plan['schedules']==[[kind,(n+1)//3-1 if kind=='correct' else n] for n in range(96) for kind in (['joint','correct'] if n<81 and n%3==2 else ['joint'])]
    assert all(not check(r) for b in batches for pair in b['pairs'].values() for r in pair)
    replay.R=R;replay.run('repair')
    worlds=read(R/'dev_worlds.json');pools=read(R.parent/'20261116_relation_solution/entity_palettes.json');native={};expected=[];expected_labels=[]
    for condition,ws in worlds.items():
        for index,w in enumerate(ws):
            cases=json.loads(json.dumps(targets(w)))
            for si,case in enumerate(cases):
                assert not check(case);role=case['observation']['role'];phase={'Planner':0,'Source':1,'Worker':2}[role]
                nonce=seed_for(w['base_group_id'],phase);seeds={k:nonce for k in ['names','facts','plans','links','options']}
                for ci,c in enumerate(case['candidates']):
                    identifier=f'{condition}_{index}_{si}_{ci}'
                    expected.append({'id':identifier,'prompt':query(case['observation'],pools['fresh'],seeds,c['action'])})
                    expected_labels.append({'id':identifier,'condition':condition,'world_index':index,'state_index':si,'candidate_index':ci,'target':c['safe_completion'],'role':role})
                if role=='Worker':
                    import hashlib
                    cid=hashlib.sha256(json.dumps(case['observation'],sort_keys=True).encode()).hexdigest();native[cid]=case
    assert rows(R/'query_observations.jsonl')==expected and rows(R/'query_labels.jsonl')==expected_labels
    obs=rows(R/'fresh_observations.jsonl');labels=rows(R/'fresh_labels.jsonl');assert len(obs)==len(labels)==144
    for o,l in zip(obs,labels):
        assert o['id']==l['id'];c=l['case'];assert c==native[o['case_id']]
        text,order=actor(c['observation'],pools['fresh'],o['seeds'],{v['action']:str(v['safe_completion']) for v in c['candidates']})
        assert text==o['prompt'] and order==o['actions'] and o['group']==('zero' if c['no_safe_completion'] else 'positive')
    for g,zero in [('zero',True),('positive',False)]:
        assert {o['case_id'] for o in obs if o['group']==g}==set(sorted(cid for cid,c in native.items() if c['no_safe_completion']==zero)[:12])
    results=[];training_times=[];generation_times=[];responses=0;initials=[]
    for arm in plan['arms']:
        target=folder/arm;cp=read(target/'CHECKPOINT_SEALED.json')
        assert all(sha(target/'adapter'/n)==h for n,h in cp['adapter_hashes'].items())
        if arm in ['one_hop','mixed_depth']:
            initials.append(read(target/'INITIAL_STATE.json'));updates=rows(target/'training.jsonl');assert len(updates)==123 and not (target/'TIMEOUT.json').exists()
            for i,(record,(kind,n)) in enumerate(zip(updates,plan['schedules'])):
                assert record['step']==record['optimizer_step']==i+1 and record['kind']==kind and record['batch_index']==n
                assert 0<record['seconds']<15 and all(math.isfinite(record[k]) for k in ['loss','gradient_norm']) and all(math.isfinite(v) for v in record['parts'].values())
                if kind=='joint':
                    b=batches[n];d=1 if arm=='one_hop' else b['mixed_depth'];assert record['depth']==d and record['key']==b['key']
                    truth=[{c['action']:str(c['safe_completion']) for c in r['candidates']} for r in b['pairs'][str(d)]]
                    assert record['actor_input_estimates']==[truth,truth]
                    assert abs(record['loss']-record['parts']['query_loss']-record['parts']['actor_loss'])<1e-5
                else:
                    correction=rows(OLD/'correction.jsonl');assert record['ids']==[c['id'] for c in correction[n*4:n*4+4]]
                    assert abs(record['loss']-sum(record['parts'].values()))<1e-5
                training_times.append(record['seconds'])
            assert cp['steps']==123 and cp['input_tokens']==sum(x['input_tokens'] for x in updates) and cp['initial_adapter_hashes']==plan['checkpoint']['seal']['adapter_hashes']
        else:assert cp==plan['earlier_checkpoint' if arm=='earlier' else 'checkpoint']['seal']
        identity=read(target/'EVAL_IDENTITY.json');assert identity['before']==identity['after'] and identity['errors']==0
        qr=rows(target/'query_raw.jsonl');assert len(qr)==len(expected)==plan['query_count'];stats=collections.defaultdict(collections.Counter);state_correct=collections.defaultdict(list)
        for i,(raw,o,l) in enumerate(zip(qr,expected,expected_labels)):
            assert raw['id']==o['id']==l['id'];found=re.findall(r'ANSWER:\s*([01])\b',raw['raw']);valid=len(found)==1 and raw['seconds_batch']<9.5
            bit=int(found[0]) if valid else None;correct=bit==l['target'];key=(l['condition'].split('_v')[0],l['role'])
            stats[key].update(queries=1,correct=int(correct),invalid=int(not valid),positive=int(l['target']==1),
                false_negative=int(bit==0 and l['target']==1),false_positive=int(bit==1 and l['target']==0))
            state_correct[l['condition'],l['world_index'],l['state_index']].append(correct)
            if i%4==0:
                assert len({r['seconds_batch'] for r in qr[i:i+4]})==1;generation_times.append(raw['seconds_batch'])
        responses+=len(qr);ar=rows(target/'fresh_raw.jsonl');assert len(ar)==144;by_case=collections.defaultdict(list);invalid=unsafe=0
        for i,(raw,o,l) in enumerate(zip(ar,obs,labels)):
            assert raw['id']==o['id']==l['id'];found=re.findall(r'ANSWER:\s*([ABC])\b',raw['raw']);valid=len(found)==1 and raw['seconds_batch']<9.5
            chosen=o['actions'][ord(found[0])-65] if valid else None;c=l['case'];best={c['candidates'][j]['action'] for j in c['optimal_indices']}
            by_case[o['case_id']].append((o['group'],tuple(o['actions']),chosen in best));invalid+=int(not valid)
            unsafe+=int(any(v['action']==chosen and not v['immediate_safe'] for v in c['candidates']))
            if i%4==0:
                assert len({r['seconds_batch'] for r in ar[i:i+4]})==1;generation_times.append(raw['seconds_batch'])
        assert all(len(v)==6 and {r[1] for r in v}==set(itertools.permutations(v[0][1])) for v in by_case.values());responses+=len(ar)
        records={c:rows(target/(c+'.jsonl')) for c in worlds};flat=[r for rs in records.values() for r in rs];timings={}
        for condition,rs in records.items():
            for index,row in enumerate(rs):
                for e in row['trace']:
                    responses+=1+len(e['queries']);key=(condition,index//2,e['phase'])
                    timings.setdefault((*key,'actor'),set()).add(e['seconds_batch'])
                    for q in e['queries']:timings.setdefault((*key,'query'),set()).add(q['seconds_batch'])
        assert all(len(v)==1 for v in timings.values());generation_times.extend(next(iter(v)) for v in timings.values())
        paired={family:sum(all(records[family+'_v'+str(v)+s][i]['strict_success'] for v in [0,1] for s in ['', '_swap']) for i in range(len(records[family+'_v0']))) for family in ['base','depth3','depth4','internal3','cycle4']}
        results.append({'arm':arm,'episodes':len(flat),'strict_success':sum(r['strict_success'] for r in flat),'violations':sum(not r['ci'] for r in flat),
            'invalid_actions':sum(r['invalid'] for r in flat),'invalid_queries':sum(r['invalid_queries'] for r in flat),'all_four_groups':paired,
            'fixed_query_count':len(qr),'fixed_query_correct':sum(v['correct'] for v in stats.values()),'fixed_all_correct_states':sum(all(v) for v in state_correct.values()),
            'fixed_states':len(state_correct),'query_by_family_role':[{'family':f,'role':r,**dict(v)} for (f,r),v in stats.items()],
            'oracle_all_six':{g:sum(all(x[2] for x in v) for v in by_case.values() if v[0][0]==g) for g in ['zero','positive']},
            'oracle_unsafe':unsafe,'oracle_invalid':invalid,'training_tokens':cp.get('input_tokens') if arm in ['one_hop','mixed_depth'] else 0})
    assert initials[0]==initials[1];lookup={r['arm']:r for r in results};m=lookup['mixed_depth'];initial=lookup['initial'];earlier=lookup['earlier']
    gate=(m['strict_success']>=max(initial['strict_success'],earlier['strict_success']) and m['violations']<=initial['violations'] and m['invalid_actions']==m['invalid_queries']==0
        and m['oracle_all_six']=={'zero':12,'positive':12} and m['oracle_unsafe']==m['oracle_invalid']==0
        and m['fixed_query_correct']>max(initial['fixed_query_correct'],lookup['one_hop']['fixed_query_correct'])
        and all(m['all_four_groups'][f]>0 for f in ['depth4','internal3','cycle4']))
    result={'arms':results,'local_gate_passed':gate,'new_updates':len(training_times),'training_seconds':sum(training_times),'max_update_seconds':max(training_times),
        'free_responses':responses,'generation_seconds':sum(generation_times),'max_generation_seconds':max(generation_times),
        'errors':0,'whole_goal_complete':False,'large_gpu':'NO_GO'}
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({**result,'arms':[{k:v for k,v in r.items() if k!='query_by_family_role'} for r in results]}))


if __name__=='__main__':main()
