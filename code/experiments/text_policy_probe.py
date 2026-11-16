"""Shared text-feature finite policy, not a language model or method result.

The learned backbone receives only hashed visible text. There are no oracle,
closure, risk, correct-answer, group, or reference features. Evaluation is only
leave-one-declared-rule-family-out within the same synthetic workflow.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import time
import zlib
import numpy as np

from kci.multi_candidates import MultiRuleCandidateGenerator
from kci.contracts import Verdict
from experiments.multi_dataset import case_env

DIM=512
KEYS=['A:SKIP::','A:SEND:context_0:','A:SEND:context_1:','A:SEND:context_0,context_1:',
      'A:NOTIFY::yes','A:NOTIFY::no','A:ABSTAIN::',
      'B:SEND:payload_0:','B:SEND:decision_yes:','B:SEND:decision_no:',
      'B:SEND:decision_yes,payload_0:','B:SEND:decision_no,payload_0:',
      'B:REQUEST_RULES::','B:COMPUTE_TASK::','B:ABSTAIN::',
      'R:SEND:task_rule:','R:FINALIZE::yes','R:FINALIZE::no','R:ABSTAIN::']


def action_class(action):
    if isinstance(action,str):action=json.loads(action)
    key=action['actor']+':'+action['kind']+':'+','.join(sorted(action['fact_ids']))+':'+action['answer']
    return KEYS.index(key)


@lru_cache(maxsize=12000)
def text_features(prompt):
    words=re.findall(r'[a-z_]+|-?\d+',prompt.lower())
    features=np.zeros(DIM,dtype=np.float64)
    for token in words+[' '.join(x) for x in zip(words,words[1:])]:
        value=zlib.crc32(token.encode('utf-8'));index=1+value%(DIM-1)
        features[index]+=1 if value&0x80000000 else -1
    norm=np.linalg.norm(features)
    if norm:features/=norm
    features[0]=1
    return features


def read_rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare(dataset,raw,output):
    out=Path(output)
    if out.exists():raise FileExistsError('new feature output required')
    out.mkdir(parents=True)
    rows=read_rows(Path(dataset)/'preferences.jsonl');sft=read_rows(Path(dataset)/'sft_controls.jsonl')
    arrays={'x':np.stack([text_features(r['prompt']) for r in rows]),
            'chosen':np.array([action_class(r['chosen']) for r in rows]),
            'rejected':np.array([action_class(r['rejected']) for r in rows]),
            'sft_x':np.stack([text_features(r['prompt']) for r in sft]),
            'sft_y':np.array([action_class(r['chosen']) for r in sft])}
    mask=np.zeros((len(sft),len(KEYS)),dtype=bool)
    for i,row in enumerate(sft):mask[i,[action_class(a) for a in row['menu']]]=True
    arrays['sft_menu']=mask
    np.savez_compressed(out/'features.npz',**arrays)
    meta={'families':[r['metadata']['rule_family_id'] for r in rows],
          'sources':[r['metadata']['training_sources'] for r in rows],
          'preference_ids':[r['metadata']['preference_id'] for r in rows],
          'sft_families':[r['rule_family_id'] for r in sft],
          'sft_observation_ids':[r['observation_class_id'] for r in sft],
          'feature_rule':'signed normalized word and adjacent-word counts; shared trainable tanh backbone',
          'hidden_label_features':False,'dim':DIM,'action_classes':KEYS,
          'inputs_sha256':{n:sha256((Path(dataset)/n).read_bytes()).hexdigest()
                          for n in ('preferences.jsonl','sft_controls.jsonl')}}
    eval_cases=[]
    for path in sorted(Path(raw).glob('*.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        initial=r['states'][0]
        costs=[x['minimum_safe_total_cost'] for x in initial['actions'] if x['minimum_safe_total_cost'] is not None]
        eval_cases.append({'spec':r['case_spec'],'optimal_cost':min(costs) if costs else None})
    meta['evaluation_cases']=eval_cases
    save_json(out/'metadata.json',meta)
    print(json.dumps({'pairs':len(rows),'sft':len(sft),'evaluation_cases':len(eval_cases)}))


def forward(x,u,e):
    h=np.tanh(x@u.T)
    return h,h@e.T


def dpo_gradient(x,c,r,u,e,reference_diff,beta=1):
    h,scores=forward(x,u,e)
    z=beta*(scores[np.arange(len(c)),c]-scores[np.arange(len(c)),r]-reference_diff)
    coefficient=-beta*np.exp(-np.logaddexp(0,z))/len(c)
    ds=np.zeros_like(scores);ds[np.arange(len(c)),c]=coefficient;ds[np.arange(len(c)),r]-=coefficient
    ge=ds.T@h;gu=((ds@e)*(1-h*h)).T@x
    return float(np.mean(np.logaddexp(0,-z))),gu,ge


def train(features,heldout,variant,seed,output):
    started=time.perf_counter();out=Path(output)
    if out.exists():raise FileExistsError('preserve prior training output')
    out.mkdir(parents=True)
    meta=json.loads((Path(features)/'metadata.json').read_text(encoding='utf-8'))
    data=np.load(Path(features)/'features.npz');x=data['x'];c=data['chosen'];r=data['rejected']
    train_ids=np.flatnonzero(np.array(meta['families'])!=heldout)
    test_ids=np.flatnonzero(np.array(meta['families'])==heldout)
    sft_ids=np.flatnonzero(np.array(meta['sft_families'])!=heldout)
    rng=np.random.default_rng(seed);u=rng.normal(0,.1,(24,DIM));e=rng.normal(0,.1,(len(KEYS),24))
    sft_batches=rng.choice(sft_ids,size=(200,16),replace=True)
    for batch in sft_batches:
        h,score=forward(data['sft_x'][batch],u,e)
        score=np.where(data['sft_menu'][batch],score,-1e30);score-=np.max(score,axis=1,keepdims=True)
        ds=np.exp(score);ds/=ds.sum(axis=1,keepdims=True);ds[np.arange(len(batch)),data['sft_y'][batch]]-=1;ds/=len(batch)
        ge=ds.T@h;gu=((ds@e)*(1-h*h)).T@data['sft_x'][batch]
        u-=.3*gu;e-=.3*ge
    ref_u=u.copy();ref_e=e.copy()
    _,ref_scores=forward(x,ref_u,ref_e);ref_diff=ref_scores[np.arange(len(c)),c]-ref_scores[np.arange(len(c)),r]
    source_pools={s:np.array([i for i in train_ids if s in meta['sources'][i]]) for s in ('ci','future','repair')}
    active=train_ids if variant in {'all','balanced','sft_only'} else source_pools[variant]
    schedule=[];losses=[]
    if variant!='sft_only':
        for step in range(200):
            pool=source_pools[('ci','future','repair')[step%3]] if variant=='balanced' else active
            batch=rng.choice(pool,size=16,replace=True);schedule.append(batch.tolist())
            loss,gu,ge=dpo_gradient(x[batch],c[batch],r[batch],u,e,ref_diff[batch])
            u-=.2*gu;e-=.2*ge
            if step%40==0:losses.append({'step':step,'loss':loss})
    check=train_ids[:4];loss,gu,ge=dpo_gradient(x[check],c[check],r[check],u,e,ref_diff[check])
    index=np.unravel_index(np.argmax(np.abs(gu)),gu.shape);eps=1e-6
    plus=u.copy();minus=u.copy();plus[index]+=eps;minus[index]-=eps
    high=dpo_gradient(x[check],c[check],r[check],plus,e,ref_diff[check])[0]
    low=dpo_gradient(x[check],c[check],r[check],minus,e,ref_diff[check])[0]
    grad_error=abs((high-low)/(2*eps)-gu[index]);assert grad_error<1e-6
    _,scores=forward(x,u,e);margin=scores[np.arange(len(c)),c]-scores[np.arange(len(c)),r]
    fit={name:{'pairs':len(ids),'accuracy':float(np.mean(margin[ids]>1e-9)),
               'by_source':{s:float(np.mean(margin[[i for i in ids if s in meta['sources'][i]]]>1e-9))
                            for s in ('ci','future','repair')}} for name,ids in (('train',train_ids),('heldout',test_ids))}
    gen=MultiRuleCandidateGenerator();episodes=[]
    for row in meta['evaluation_cases']:
        if row['spec']['family']!=heldout:continue
        env=case_env(row['spec']);state=env.initial();actions=[];status='TASK_FAILURE'
        while not state.done:
            obs=env.observation(state,state.actor);menu=gen.generate(obs,12).actions
            _,score=forward(text_features(obs.as_prompt())[None],u,e)
            ids=[action_class(a.__dict__) for a in menu];action=menu[int(np.argmax(score[0,ids]))]
            after,audit=env.transition(state,action)
            actions.append({'action':action.as_json(),'verdict':audit.verdict.value});state=after
            if audit.verdict!=Verdict.VALID:
                status='VIOLATION' if audit.verdict==Verdict.VIOLATION else 'UNKNOWN';break
        else:status='SAFE_SUCCESS' if state.success and not state.any_violation else 'TASK_FAILURE'
        episodes.append({'case':row['spec']['case_id'],'status':status,'cost':state.cost,'optimal_cost':row['optimal_cost'],'actions':actions})
    weights=out/'weights.npz';np.savez_compressed(weights,u=u,e=e,reference_u=ref_u,reference_e=ref_e)
    result={'heldout_family':heldout,'variant':variant,'seed':seed,'sft_steps':200,
            'dpo_steps':len(schedule),'batch_size':16,'sft_lr':.3,'dpo_lr':.2,'beta':1,
            'parameters':u.size+e.size,'shared_backbone':True,'gradient_error':grad_error,
            'fit':fit,'episodes':episodes,'episode_counts':dict(Counter(r['status'] for r in episodes)),
            'batch_schedule':schedule,'sft_batches_sha256':sha256(sft_batches.tobytes()).hexdigest(),
            'loss_trace':losses,'weights_sha256':sha256(weights.read_bytes()).hexdigest(),
            'feature_metadata_sha256':sha256((Path(features)/'metadata.json').read_bytes()).hexdigest(),
            'seconds':time.perf_counter()-started,
            'scope':'shared-workflow family-heldout finite text-feature policy, NOT LLM or independent-task generalization'}
    save_json(out/'result.json',result)
    print(json.dumps({k:result[k] for k in ('heldout_family','variant','seed','fit','episode_counts','seconds')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--dataset');p.add_argument('--raw')
    p.add_argument('--features');p.add_argument('--heldout');p.add_argument('--variant');p.add_argument('--seed',type=int)
    p.add_argument('--output',required=True);a=p.parse_args()
    if a.prepare:prepare(a.dataset,a.raw,a.output)
    else:train(a.features,a.heldout,a.variant,a.seed,a.output)
