"""Reconstruct native actions, loss inputs, and oracle permutation diagnostics."""
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
from weighting import weights
from worlds import targets
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
    assert read(R/'IDENTITY_CHECK.json')['errors']==0
    assert all(v<1e-6 for v in read(R/'IDENTITY_CHECK.json')['deltas'].values())
    replay.R=R;replay.run('repair')
    summaries=read(folder/'RESULTS.json')['summaries'];data=buckets();timings=[];tokens={};metrics=[];diagnostics=[]
    generation_times=[];responses=0
    train_obs=rows(R.parent/'20261117_option_fallback/observations.jsonl')
    train_labels=rows(R.parent/'20261117_option_fallback/labels.jsonl')
    original={c['case_id']:c for c in rows(R.parent/'20261117_frozen_crossover/train_oracle.jsonl')}
    prior=rows(R.parent/'20261117_option_fallback/responses.jsonl')
    fresh_obs=rows(R/'fresh_observations.jsonl');fresh_labels=rows(R/'fresh_labels.jsonl')
    pools=read(R.parent/'20261116_relation_solution/entity_palettes.json')
    candidates={}
    for ws in read(R/'dev_worlds.json').values():
        for w in ws:
            for c in targets(w):
                if c['observation']['role']=='Worker':
                    cid=hashlib.sha256(json.dumps(c['observation'],sort_keys=True).encode()).hexdigest();candidates[cid]=c
    for group,zero in [('zero',True),('positive',False)]:
        expected=sorted(cid for cid,c in candidates.items() if (not any(x['safe_completion'] for x in c['candidates']))==zero)[:12]
        assert set(expected)=={o['case_id'] for o in fresh_obs if o['group']==group}
    for arm in plan['arms']:
        reports=[r for r in summaries if r['arm']==arm]
        aggregate={k:sum(r[k] for r in reports) for k in ['episodes','strict_success','valid_ci','invalid','invalid_queries']}
        paired={}
        for family in ['base','depth3','internal3','cycle4']:
            groups=[rows(folder/arm/(family+'_v'+str(v)+suffix+'.jsonl')) for v in (0,1) for suffix in ['', '_swap']]
            paired[family]={'groups':len(groups[0]),'all_four_success':sum(all(r['strict_success'] for r in rs) for rs in zip(*groups))}
        metrics.append({'arm':arm,**aggregate,'paired':paired})
        for report in reports:
            episodes=rows(folder/arm/(report['condition']+'.jsonl'))
            responses+=sum(1+len(e['queries']) for r in episodes for e in r['trace'])
            for phase in range(3):
                for start in range(0,len(episodes),2):
                    es=[r['trace'][phase] for r in episodes[start:start+2] if len(r['trace'])>phase]
                    if es:
                        assert len({e['seconds_batch'] for e in es})==len({q['seconds_batch'] for e in es for q in e['queries']})==1
                        generation_times.extend([es[0]['seconds_batch'],es[0]['queries'][0]['seconds_batch']])
        if arm in plan['multipliers']:
            updates=rows(folder/arm/'training.jsonl');assert len(updates)==80 and not (folder/arm/'TIMEOUT.json').exists()
            all_weights=[]
            for step,(b,r) in enumerate(zip(plan['batches'],updates)):
                assert r['step']==step+1 and r['role']==b['key'][0] and r['multiplier']==plan['multipliers'][arm]
                assert 0<r['seconds']<12 and all(math.isfinite(r[k]) for k in ['gradient_norm','query_loss','actor_loss'])
                pair=data[tuple(b['key'])][b['index']];truth=[{c['action']:str(c['safe_completion']) for c in row['candidates']} for row in pair]
                assert r['actor_input_estimates']==[truth,truth] and r['actor_js']==0
                all_weights.extend(weights(pair,r['multiplier'])*2);timings.append(r['seconds'])
            assert len(all_weights)==320 and abs(sum(all_weights)-320)<1e-8
            tokens[arm]=sum(r['input_tokens'] for r in updates)
            cp=read(folder/arm/'CHECKPOINT_SEALED.json')
            assert cp['input_tokens']==tokens[arm] and cp['steps']==80 and cp['initial_adapter_hashes']==plan['checkpoint']['seal']['adapter_hashes']
        else:assert read(folder/arm/'CHECKPOINT_SEALED.json')==plan['checkpoint']['seal']
        for name,observations,labels in [('train',train_obs,train_labels),('fresh',fresh_obs,fresh_labels)]:
            generated=rows(folder/arm/(name+'_orders.jsonl'));assert len(generated)==len(observations)==len(labels)
            stats=defaultdict(Counter);by_case=defaultdict(list);repeat=0
            for i,(obs,label,row) in enumerate(zip(observations,labels,generated)):
                assert row['id']==obs['id']==label['id']
                case=original[obs['case_id']] if name=='train' else label['case']
                seeds={**case['seeds'],'options':obs['options_seed']} if name=='train' else obs['seeds']
                if name=='fresh':assert case['observation']==candidates[obs['case_id']]['observation'] and case['candidates']==candidates[obs['case_id']]['candidates']
                bits={c['action']:str(c['safe_completion']) for c in case['candidates']}
                text,order=actor(case['observation'],pools['train' if name=='train' else 'fresh'],seeds,bits)
                assert text==obs['prompt'] and order==obs['actions']
                assert obs['group']==('zero' if not any(c['safe_completion'] for c in case['candidates']) else 'positive')
                best={case['candidates'][j]['action'] for j in case['optimal_indices']}
                found=re.findall(r'ANSWER:\s*([ABC])\b',row['raw']);valid=len(found)==1 and row['seconds_batch']<9.5
                chosen=order[ord(found[0])-65] if valid else None
                candidate=next((c for c in case['candidates'] if c['action']==chosen),None)
                unsafe=candidate is not None and not candidate['immediate_safe']
                stats[(obs['group'],order.index('PUBLISH_FULL'))].update(n=1,optimal=int(chosen in best),unsafe=int(unsafe),invalid=int(not valid))
                by_case[obs['case_id']].append((obs['group'],tuple(order),chosen in best))
                if name=='train' and arm=='initial':repeat+=row['raw']!=prior[i]['raw']
                if i%4==0:
                    assert len({r['seconds_batch'] for r in generated[i:i+4]})==1;generation_times.append(row['seconds_batch'])
            for values in by_case.values():assert {v[1] for v in values}==set(itertools.permutations(values[0][1])) and len(values)==6
            diagnostics.append({'arm':arm,'split':name,'positions':[{'group':g,'full_position':'ABC'[p],**dict(v)} for (g,p),v in stats.items()],
                'all_six':{g:{'cases':sum(vs[0][0]==g for vs in by_case.values()),'optimal':sum(all(v[2] for v in vs) for vs in by_case.values() if vs[0][0]==g)} for g in ['zero','positive']},
                'identical_checkpoint_raw_disagreements':repeat if name=='train' and arm=='initial' else None})
            responses+=len(generated)
    by_arm={r['arm']:r for r in metrics};fresh={r['arm']:r for r in diagnostics if r['split']=='fresh'}
    passed=all(by_arm['weighted'][k]>=by_arm[a][k] for a in ['initial','uniform'] for k in ['strict_success','valid_ci'])
    passed&=all(fresh['weighted']['all_six']['zero']['optimal']>fresh[a]['all_six']['zero']['optimal'] for a in ['initial','uniform'])
    passed&=sum(r['unsafe']+r['invalid'] for r in fresh['weighted']['positions'] if r['group']=='zero')==0
    result={'dev':metrics,'oracle_diagnostics':diagnostics,'pilot_gate':bool(passed),'new_updates':len(timings),'training_seconds':sum(timings),
        'max_update_seconds':max(timings),'tokens':tokens,'responses':responses,'generation_seconds':sum(generation_times),
        'max_generation_seconds':max(generation_times),'errors':0,'large_gpu':'NO_GO','whole_goal_complete':False}
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
