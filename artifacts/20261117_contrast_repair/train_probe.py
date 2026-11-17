"""Frozen in-sample relation fit and same-world entity-renaming diagnostic."""
import collections
import gc
import json
import re
import sys
from paths import R,ROOT,PARENT,read,rows,sha,verify_prior
from render import query


def cases():
    pools=read(R.parent/'20261116_relation_solution/entity_palettes.json');obs=[];labels=[]
    for arm in ['one_hop','mixed_depth']:
        for b in rows(R/'train_batches.jsonl'):
            if not b['forced_contrast']:continue
            depth=1 if arm=='one_hop' else b['mixed_depth'];pair=b['pairs'][str(depth)]
            for view,seeds in enumerate(b['views']):
                for ci,(a,c) in enumerate(zip(pair[0]['candidates'],pair[1]['candidates'])):
                    if a['safe_completion']==c['safe_completion']:continue
                    pid=f"{arm}_{b['step']}_{view}_{ci}"
                    for vocab,poolname in [('original','train'),('renamed','fresh')]:
                        for side,row in enumerate(pair):
                            candidate=row['candidates'][ci];identifier=f'{pid}_{vocab}_{side}'
                            obs.append({'id':identifier,'arm':arm,'pair_id':pid,'vocab':vocab,'side':side,
                                'prompt':query(row['observation'],pools[poolname],seeds,candidate['action'])})
                            labels.append({'id':identifier,'target':candidate['safe_completion'],'role':b['key'][0],'depth':depth,'batch':b['step']})
    return obs,labels


def build():
    assert not (R/'TRAIN_PROBE_PLAN.json').exists();assert not (R/'SEALED.json').exists()
    obs,labels=cases();assert len(obs)==len(labels)==1152
    for name,data in [('train_probe_observations.jsonl',obs),('train_probe_labels.jsonl',labels)]:
        (R/name).write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')
    plan={'scope':'Supplementary diagnostic registered during repair training, before completed repair results. Original TRAIN prompts and same-world renamed prompts; in-sample fit and symbol transfer, not independent generalization or full policy success.',
        'source_sha256':sha(R/'train_probe.py'),'main_plan_sha256':sha(R/'PLAN.json'),
        'data_hashes':{n:sha(R/n) for n in ['train_probe_observations.jsonl','train_probe_labels.jsonl']},
        'responses':1152,'new_updates':0,'primary_gate_changed':False,
        'rename_interpretation_gate':'All four prompts in each original/renamed relationship pair have identical token counts. Preserve and report all outputs even if gate fails.'}
    (R/'TRAIN_PROBE_PLAN.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))


def verify():
    verify_prior();plan=read(R/'PLAN.json');probe=read(R/'TRAIN_PROBE_PLAN.json');seal=read(R/'SEALED.json')
    assert seal['complete'] and all(sha(R/n)==h for n,h in seal['files'].items())
    assert all(sha(R/n)==h for n,h in {**plan['source_hashes'],**plan['data_hashes']}.items())
    assert all(sha(ROOT/n)==h for n,h in plan['dependencies'].items())
    assert sha(R/'PLAN.json')==probe['main_plan_sha256'] and sha(R/'train_probe.py')==probe['source_sha256']
    assert all(sha(R/n)==h for n,h in probe['data_hashes'].items())
    return probe


def run():
    plan=verify();assert not (R/'train_probe_raw.jsonl').exists()
    sys.path.insert(0,str(R.parent/'20261117_frozen_crossover'))
    from common import load,generate,verify as verify_model
    verify_model();obs=rows(R/'train_probe_observations.jsonl');count=0
    for arm in ['one_hop','mixed_depth']:
        folder=R/'repair'/arm;cp={'path':folder.relative_to(ROOT).as_posix(),'seal':read(folder/'CHECKPOINT_SEALED.json')}
        tokenizer,model=load('fixed',{'checkpoints':{'fixed':cp}});selected=[o for o in obs if o['arm']==arm]
        for start in range(0,len(selected),4):
            batch=selected[start:start+4];assert len({o['pair_id'] for o in batch})==1
            lengths=[len(tokenizer.apply_chat_template([{'role':'user','content':o['prompt']}],tokenize=True,add_generation_prompt=True)) for o in batch]
            raw,seconds=generate(tokenizer,model,[o['prompt'] for o in batch])
            with (R/'train_probe_raw.jsonl').open('a',encoding='utf-8') as f:
                for o,value,length in zip(batch,raw,lengths):f.write(json.dumps({'id':o['id'],'raw':value,'prompt_tokens':length,'seconds_batch':seconds})+'\n');count+=1
        print(json.dumps({'arm':arm,'cumulative_responses':count}),flush=True);del model;gc.collect()
    assert count==plan['responses']
    (R/'TRAIN_PROBE_SEALED.json').write_text(json.dumps({'responses':count,'raw_sha256':sha(R/'train_probe_raw.jsonl'),'plan_sha256':sha(R/'TRAIN_PROBE_PLAN.json'),'main_seal_sha256':sha(R/'SEALED.json'),'model_updates':0},indent=2))


def audit():
    plan=verify();seal=read(R/'TRAIN_PROBE_SEALED.json');assert sha(R/'TRAIN_PROBE_PLAN.json')==seal['plan_sha256']
    assert sha(R/'SEALED.json')==seal['main_seal_sha256'] and sha(R/'train_probe_raw.jsonl')==seal['raw_sha256']
    obs,labels=cases();assert obs==rows(R/'train_probe_observations.jsonl') and labels==rows(R/'train_probe_labels.jsonl')
    raw=rows(R/'train_probe_raw.jsonl');assert len(raw)==len(obs)==seal['responses']==1152
    stats=collections.defaultdict(collections.Counter);paired=collections.defaultdict(list);lengths=collections.defaultdict(list);times=[]
    for i,(o,l,r) in enumerate(zip(obs,labels,raw)):
        assert o['id']==l['id']==r['id'];found=re.findall(r'ANSWER:\s*([01])\b',r['raw']);valid=len(found)==1 and r['seconds_batch']<9.5
        correct=valid and int(found[0])==l['target'];key=(o['arm'],o['vocab'],l['role'],l['depth'])
        stats[key].update(queries=1,correct=int(correct),invalid=int(not valid));paired[key,o['pair_id']].append((o['side'],l['target'],correct))
        lengths[o['pair_id']].append(r['prompt_tokens'])
        if i%4==0:
            assert len({v['seconds_batch'] for v in raw[i:i+4]})==1;times.append(r['seconds_batch'])
    for (key,pid),v in paired.items():
        assert len(v)==2 and {x[0] for x in v}=={0,1} and {x[1] for x in v}=={0,1}
        stats[key].update(pairs=1,both_correct=int(all(x[2] for x in v)))
    assert len(lengths)==288 and all(len(v)==4 for v in lengths.values())
    result={'summaries':[{'arm':a,'vocab':v,'role':r,'depth':d,**dict(s)} for (a,v,r,d),s in stats.items()],
        'rename_token_controls_passed':all(len(set(v))==1 for v in lengths.values()),'token_control_groups':len(lengths),
        'responses':1152,'generation_seconds':sum(times),'max_seconds':max(times),'model_updates':0,'errors':0,'scope':plan['scope'],
        'primary_gate_changed':False,'whole_goal_complete':False}
    (R/'TRAIN_PROBE_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':{'build':build,'run':run,'audit':audit}[sys.argv[1]]()
