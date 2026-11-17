"""Replay all policies and validate matched updates and original TRAIN-fit prompts."""
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(R.parent/'20261117_teacher_binding'), str(R.parent/'20261117_sampler_repair')]
import replay
from sampling import buckets
sys.path.insert(0,str(R.parent/'20261117_binding_factors'))
from render import query, actor

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
rows = lambda p: list(map(json.loads,p.read_text(encoding='utf-8').splitlines()))


def main():
    plan = read(R/'PLAN.json'); folder=R/'repair'; seal=read(R/'SEALED.json')
    assert read(folder/'PLAN.json') == plan and seal['complete']
    assert all(sha(R/p)==h for p,h in seal['files'].items())
    assert all(sha(R/p)==h for p,h in plan['source_hashes'].items())
    assert all(sha(ROOT/p)==h for p,h in plan['dependencies'].items())
    assert sha(R/'dev_worlds.json') == sha(folder/'dev_worlds.json') == plan['dev_sha256']
    assert plan['batches']==read(R.parent/'20261117_teacher_binding/PLAN.json')['batches']
    replay.R=R; replay.run('repair')
    summaries=read(folder/'RESULTS.json')['summaries']; worlds=read(R/'dev_worlds.json')
    metrics=[]; generation_seconds=[]; generation_responses=0
    for arm in plan['arms']:
        reports=[row for row in summaries if row['arm']==arm]
        aggregate={key:sum(row[key] for row in reports) for key in ['episodes','strict_success','valid_ci','invalid','invalid_queries']}
        paired={}
        for family in ['base','depth3','internal3','cycle4']:
            groups=[rows(folder/arm/(family+'_v'+str(v)+suffix+'.jsonl')) for v in (0,1) for suffix in ['', '_swap']]
            assert len({len(group) for group in groups})==1
            for a,b,c,d in zip(*groups):
                assert a['world_id']==c['world_id'] and b['world_id']==d['world_id']
            paired[family]={'groups':len(groups[0]),'all_four_success':sum(all(r['strict_success'] for r in group) for group in zip(*groups))}
        metrics.append({'arm':arm,**aggregate,'paired':paired})
        for condition in worlds:
            records=rows(folder/arm/(condition+'.jsonl'))
            generation_responses+=sum(1+len(entry['queries']) for row in records for entry in row['trace'])
            for phase in range(3):
                for start in range(0,len(records),2):
                    entries=[row['trace'][phase] for row in records[start:start+2] if len(row['trace'])>phase]
                    if entries:
                        assert len({e['seconds_batch'] for e in entries})==1
                        assert len({q['seconds_batch'] for e in entries for q in e['queries']})==1
                        generation_seconds.extend([entries[0]['seconds_batch'], entries[0]['queries'][0]['seconds_batch']])
        if arm not in plan['rates']:
            assert read(folder/arm/'CHECKPOINT_SEALED.json')==plan['checkpoints'][arm]['seal']
    train=rows(R.parent/'20261117_frozen_crossover/train_oracle.jsonl'); data=buckets()
    pool=read(R.parent/'20261116_relation_solution/entity_palettes.json')['train']
    fit=defaultdict(Counter); training_seconds=[]; training_tokens={}
    for arm,rate in plan['rates'].items():
        updates=rows(folder/arm/'training.jsonl'); assert len(updates)==80
        assert not (folder/arm/'TIMEOUT.json').exists()
        for step,(batch,row) in enumerate(zip(plan['batches'],updates)):
            assert row['step']==step+1 and row['role']==batch['key'][0] and row['lr']==rate
            assert row['seconds']<9.5 and row['actor_js']==0
            assert all(math.isfinite(row[k]) for k in ['query_loss','actor_loss','gradient_norm_before_clip'])
            pair=data[tuple(batch['key'])][batch['index']]
            assert not [e for case in pair for e in replay.check(case)]
            truth=[{c['action']:str(c['safe_completion']) for c in case['candidates']} for case in pair]
            assert row['actor_input_estimates']==[truth,truth]
            training_seconds.append(row['seconds'])
        training_tokens[arm]=sum(row['input_tokens'] for row in updates)
        checkpoint=read(folder/arm/'CHECKPOINT_SEALED.json')
        assert checkpoint['steps']==80 and checkpoint['input_tokens']==training_tokens[arm]
        assert checkpoint['initial_adapter_hashes']==plan['checkpoints']['teacher_sft']['seal']['adapter_hashes']
        records=rows(folder/arm/'train_fit.jsonl'); assert len(records)==len(train)==320
        for i,(case,row) in enumerate(zip(train,records)):
            assert row['case_id']==case['case_id']
            obs=case['observation']; truth={c['action']:str(c['safe_completion']) for c in case['candidates']}
            assert [q['action'] for q in row['queries']]==obs['actions']
            stats=fit[arm,obs['role']]; good=True
            for q in row['queries']:
                assert q['prompt']==query(obs,pool,case['seeds'],q['action'])
                matches=re.findall(r'ANSWER:\s*([01])\b',q['raw']); valid=len(matches)==1 and q['seconds_batch']<9.5
                bit=matches[0] if valid else 'unknown'
                assert q['valid']==valid and q['estimate']==bit
                correct=bit==truth[q['action']];good &= correct
                stats.update(queries=1,query_correct=int(correct),invalid_queries=int(not valid))
            actual=row['actor']; text,order=actor(obs,pool,case['seeds'],truth)
            assert actual['prompt']==text and actual['actions']==order and actual['input_estimates']==truth
            matches=re.findall(r'ANSWER:\s*([ABC])\b',actual['raw'])
            valid=len(matches)==1 and ord(matches[0])-65<len(order) and actual['seconds_batch']<9.5
            selected=order[ord(matches[0])-65] if valid else None
            assert actual['valid']==valid and actual['selected']==selected
            best={case['candidates'][j]['action'] for j in case['optimal_indices']}
            stats.update(presentations=1,all_queries_correct=int(good),actor_optimal=int(selected in best),invalid_actions=int(not valid))
            generation_responses+=1+len(row['queries'])
            if i%4==0:
                batch=records[i:i+4]
                assert len({r['actor']['seconds_batch'] for r in batch})==1
                assert len({q['seconds_batch'] for r in batch for q in r['queries']})==1
                generation_seconds.extend([actual['seconds_batch'],row['queries'][0]['seconds_batch']])
    result={'dev':metrics,'train_fit':[{'arm':a,'role':r,**dict(v)} for (a,r),v in fit.items()],
            'new_optimizer_updates':len(training_seconds),'training_input_tokens':training_tokens,
            'max_update_seconds':max(training_seconds),'training_seconds_sum':sum(training_seconds),
            'generated_responses':generation_responses,'generation_batches':len(generation_seconds),
            'max_generation_seconds':max(generation_seconds),'generation_seconds_sum':sum(generation_seconds),
            'errors':0,'whole_goal_complete':False}
    assert result['new_optimizer_updates']==160 and sum(r['episodes'] for r in metrics)==448
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
