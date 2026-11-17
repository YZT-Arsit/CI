"""Native replay and paired diagnostics for the frozen order intervention."""
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter,defaultdict
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path[:0]=[str(R.parent/'20261117_teacher_binding'),str(R.parent/'20261117_sampler_repair'),str(R.parent/'20261117_binding_factors')]
import replay
from sampling import buckets
from render import actor
from worlds import targets
from schedules import make_schedule
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))


def main():
    plan=read(R/'PLAN.json');folder=R/'repair';seal=read(R/'SEALED.json')
    assert seal['complete'] and all(sha(R/n)==h for n,h in seal['files'].items())
    assert all(sha(R/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
    assert read(folder/'PLAN.json')==plan and read(folder/'dev_worlds.json')==read(R/'dev_worlds.json')
    assert plan['batches']==read(R.parent/'20261117_teacher_binding/PLAN.json')['batches']
    replay.R=R;replay.run('repair')
    correction=rows(R/'correction.jsonl');data=buckets();pools=read(R.parent/'20261116_relation_solution/entity_palettes.json')
    train_obs=rows(R.parent/'20261117_option_fallback/observations.jsonl');train_labels=rows(R.parent/'20261117_option_fallback/labels.jsonl')
    original={r['case_id']:r for r in rows(R.parent/'20261117_frozen_crossover/train_oracle.jsonl')}
    for c,o,l in zip(correction,train_obs,train_labels):
        assert o['group']=='zero' and c['id']==o['id']==l['id'] and c['prompt']==o['prompt'] and c['actions']==o['actions']
        good=[i for i,a in enumerate(o['actions']) if a in l['optimal_actions']]
        choice=good[int(hashlib.sha256(o['id'].encode()).hexdigest()[:8],16)%2]
        assert c['good_indices']==good and c['answer']=='ANSWER:\n'+'ABC'[choice]
    assert len(correction)==108
    fresh_obs=rows(R/'fresh_observations.jsonl');fresh_labels=rows(R/'fresh_labels.jsonl');native={}
    for ws in read(R/'dev_worlds.json').values():
        for w in ws:
            for c in json.loads(json.dumps(targets(w))):
                if c['observation']['role']=='Worker':
                    native[hashlib.sha256(json.dumps(c['observation'],sort_keys=True).encode()).hexdigest()]=c
    for group,zero in [('zero',True),('positive',False)]:
        selected=sorted(cid for cid,c in native.items() if (not any(x['safe_completion'] for x in c['candidates']))==zero)[:12]
        assert set(selected)=={o['case_id'] for o in fresh_obs if o['group']==group}
    training_times=[];generation_times=[];responses=0;metrics=[];diagnostics=[];state_records={};initial_states=[];token_counts={}
    for arm in plan['arms']:
        updates=[]
        if arm in plan['schedules']:
            schedule=make_schedule(arm);assert schedule==plan['schedules'][arm]
            updates=rows(folder/arm/'training.jsonl');assert len(updates)==107 and not (folder/arm/'TIMEOUT.json').exists()
            initial_states.append(read(folder/arm/'INITIAL_STATE.json'))
            tokens=0
            for step,((kind,n),row) in enumerate(zip(schedule,updates)):
                assert row['step']==row['optimizer_step']==step+1 and row['kind']==kind and row['batch_index']==n
                assert 0<row['seconds']<12 and all(math.isfinite(row[k]) for k in ['gradient_norm','loss'])
                assert all(math.isfinite(v) for v in row['parts'].values())
                if kind=='correct':
                    assert row['ids']==[c['id'] for c in correction[n*4:n*4+4]]
                    assert abs(row['loss']-sum(row['parts'].values()))<1e-5
                else:
                    b=plan['batches'][n];pair=data[tuple(b['key'])][b['index']]
                    truth=[{c['action']:str(c['safe_completion']) for c in x['candidates']} for x in pair]
                    assert row['actor_input_estimates']==[truth,truth]
                    assert abs(row['loss']-row['parts']['query_loss']-row['parts']['actor_loss'])<1e-5
                token_counts.setdefault((kind,n),set()).add(row['input_tokens']);tokens+=row['input_tokens'];training_times.append(row['seconds'])
            cp=read(folder/arm/'CHECKPOINT_SEALED.json');assert cp['steps']==107 and cp['input_tokens']==tokens and cp['initial_adapter_hashes']==plan['checkpoint']['seal']['adapter_hashes']
        else:assert read(folder/arm/'CHECKPOINT_SEALED.json')==plan['checkpoint']['seal']
        stages=['final']+(['midpoint'] if arm in plan['milestones'] else [])
        for stage in stages:
            target=folder/arm if stage=='final' else folder/arm/'midpoint'
            cp=read(target/'CHECKPOINT_SEALED.json');assert all(sha(target/'adapter'/n)==h for n,h in cp['adapter_hashes'].items())
            if stage=='midpoint':
                m=plan['milestones'][arm];assert cp['steps']==m and cp['input_tokens']==sum(r['input_tokens'] for r in updates[:m])
            identity=read(target/'EVAL_IDENTITY.json');assert identity['before']==identity['after'] and identity['errors']==0
            for split,observations,labels in [('train',train_obs,train_labels),('fresh',fresh_obs,fresh_labels)]:
                generated=rows(target/(split+'_orders.jsonl'));assert len(generated)==len(observations)==len(labels)
                stats=defaultdict(Counter);by_case=defaultdict(list);by_id={}
                for i,(o,l,r) in enumerate(zip(observations,labels,generated)):
                    assert o['id']==l['id']==r['id']
                    case=original[o['case_id']] if split=='train' else l['case']
                    if split=='fresh':assert all(case[k]==native[o['case_id']][k] for k in ['observation','candidates','optimal_indices','source_hashes','no_safe_completion'])
                    seeds={**case['seeds'],'options':o['options_seed']} if split=='train' else o['seeds']
                    bits={c['action']:str(c['safe_completion']) for c in case['candidates']}
                    text,order=actor(case['observation'],pools['train' if split=='train' else 'fresh'],seeds,bits)
                    assert text==o['prompt'] and order==o['actions']
                    assert o['group']==('zero' if not any(c['safe_completion'] for c in case['candidates']) else 'positive')
                    found=re.findall(r'ANSWER:\s*([ABC])\b',r['raw']);valid=len(found)==1 and r['seconds_batch']<9.5
                    chosen=order[ord(found[0])-65] if valid else None
                    good={case['candidates'][j]['action'] for j in case['optimal_indices']}
                    candidate=next((c for c in case['candidates'] if c['action']==chosen),None);unsafe=candidate is not None and not candidate['immediate_safe']
                    stats[(o['group'],order.index('PUBLISH_FULL'))].update(n=1,optimal=int(chosen in good),unsafe=int(unsafe),invalid=int(not valid))
                    by_case[o['case_id']].append((o['group'],tuple(order),chosen in good));by_id[o['id']]={'optimal':chosen in good,'unsafe':unsafe,'group':o['group']}
                    if i%4==0:
                        assert len({v['seconds_batch'] for v in generated[i:i+4]})==1;generation_times.append(r['seconds_batch'])
                for vs in by_case.values():assert len(vs)==6 and {v[1] for v in vs}==set(itertools.permutations(vs[0][1]))
                diagnostics.append({'arm':arm,'stage':stage,'split':split,'positions':[{'group':g,'full_position':'ABC'[p],**dict(s)} for (g,p),s in stats.items()],
                    'all_six':{g:{'cases':sum(vs[0][0]==g for vs in by_case.values()),'optimal':sum(all(v[2] for v in vs) for vs in by_case.values() if vs[0][0]==g)} for g in ['zero','positive']}})
                state_records[arm,stage,split]=by_id;responses+=len(generated)
        records={c:rows(folder/arm/(c+'.jsonl')) for c in read(R/'dev_worlds.json')};flat=[r for rs in records.values() for r in rs]
        paired={}
        for family in ['base','depth3','internal3','cycle4']:
            groups=[records[family+'_v'+str(v)+suffix] for v in (0,1) for suffix in ['', '_swap']]
            paired[family]={'groups':len(groups[0]),'all_four_success':sum(all(x['strict_success'] for x in xs) for xs in zip(*groups))}
        metrics.append({'arm':arm,'episodes':len(flat),'strict_success':sum(r['strict_success'] for r in flat),'violations':sum(not r['ci'] for r in flat),
                        'invalid':sum(r['invalid'] for r in flat),'invalid_queries':sum(r['invalid_queries'] for r in flat),'paired':paired})
        responses+=sum(1+len(e['queries']) for r in flat for e in r['trace'])
        for rs in records.values():
            for phase in range(3):
                for start in range(0,len(rs),2):
                    es=[r['trace'][phase] for r in rs[start:start+2] if len(r['trace'])>phase]
                    if es:
                        assert len({e['seconds_batch'] for e in es})==len({q['seconds_batch'] for e in es for q in e['queries']})==1
                        generation_times.extend([es[0]['seconds_batch'],es[0]['queries'][0]['seconds_batch']])
    assert initial_states[0]==initial_states[1]==initial_states[2] and all(len(v)==1 for v in token_counts.values())
    transitions=[]
    for arm in plan['milestones']:
        for split in ['train','fresh']:
            a=state_records['initial','final',split];b=state_records[arm,'midpoint',split];c=state_records[arm,'final',split]
            for group in ['zero','positive']:
                ids=[k for k in a if a[k]['group']==group]
                transitions.append({'arm':arm,'split':split,'group':group,'n':len(ids),
                    'initial_correct':sum(a[k]['optimal'] for k in ids),'mid_correct':sum(b[k]['optimal'] for k in ids),'final_correct':sum(c[k]['optimal'] for k in ids),
                    'initial_wrong_mid_correct':sum(not a[k]['optimal'] and b[k]['optimal'] for k in ids),
                    'mid_correct_final_wrong':sum(b[k]['optimal'] and not c[k]['optimal'] for k in ids),
                    'initial_wrong_mid_correct_final_wrong':sum(not a[k]['optimal'] and b[k]['optimal'] and not c[k]['optimal'] for k in ids)})
    by_arm={m['arm']:m for m in metrics};diag={d['arm']:d for d in diagnostics if d['stage']=='final' and d['split']=='fresh'}
    chosen=diag['interleaved'];gate=(chosen['all_six']['zero']['optimal']==12 and sum(p['unsafe']+p['invalid'] for p in chosen['positions'] if p['group']=='zero')==0
        and chosen['all_six']['positive']['optimal']>=diag['initial']['all_six']['positive']['optimal']
        and by_arm['interleaved']['strict_success']>=by_arm['initial']['strict_success'] and by_arm['interleaved']['violations']<=by_arm['initial']['violations'])
    result={'dev':metrics,'diagnostics':diagnostics,'paired_transitions':transitions,'interleaved_local_gate':bool(gate),'new_updates':len(training_times),
            'training_seconds':sum(training_times),'max_update_seconds':max(training_times),'tokens_per_arm':sum(next(iter(v)) for v in token_counts.values()),
            'free_responses':responses,'generation_seconds':sum(generation_times),'max_generation_seconds':max(generation_times),
            'errors':0,'large_gpu':'NO_GO','whole_goal_complete':False}
    assert len(training_times)==321
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
