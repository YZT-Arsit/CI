"""Small shared linear text policy; horizon transfer, not independent-task learning."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from collections import Counter
from hashlib import sha256
import argparse,json
from pathlib import Path
import numpy as np
from experiments.text_policy_probe import text_features
from kci.dynamic_tasks import DynamicTask,canonical


def run(data,source,output,variant,seed):
    data=Path(data);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    states=[json.loads(s) for s in (data/'states.jsonl').read_text(encoding='utf-8').splitlines()]
    pairs=[json.loads(s) for s in (data/'preferences.jsonl').read_text(encoding='utf-8').splitlines()]
    actions=sorted({a['action'] for s in states for a in s['actions']});idx={a:i for i,a in enumerate(actions)}
    long=lambda r:r['case_id'].endswith(('_t9','_t10'))
    primary=lambda r:json.loads(r['prompt'])['phase']=='choose' and '_a0_' in r['case_id']
    sft=[]
    for s in states:
        if not long(s) or s['state']['phase']=='choose':continue
        feasible=[a for a in s['actions'] if a['future'] and a['future']['min_total_cost'] is not None]
        if feasible:
            a=min(feasible,key=lambda a:a['future']['min_total_cost'])['action']
            sft.append((text_features(s['prompt']),idx[a],[idx[a['action']] for a in s['actions']]))
    selected=[r for r in pairs if long(r) and
       (variant=='ci' and r['level']=='immediate_CI' or variant=='history' and primary(r) or
        variant=='ordinary' and r['level'].startswith('future') and json.loads(r['prompt'])['phase']!='choose')]
    rng=np.random.default_rng(seed);w=rng.normal(0,.01,(len(actions),512))
    for step in range(200):
        gradient=np.zeros_like(w)
        for row in rng.integers(0,len(sft),8):
            x,chosen,legal=sft[row];logits=w[legal]@x;probs=np.exp(logits-max(logits));probs/=sum(probs)
            gradient[legal]+=probs[:,None]*x[None,:];gradient[chosen]-=x
        w-=.3*gradient/8
    reference=w.copy();ref_hash=sha256(reference.tobytes()).hexdigest();loss=[]
    if variant!='sft_only':
        assert selected
        x=np.stack([text_features(r['prompt']) for r in selected]);c=np.array([idx[r['chosen']] for r in selected]);b=np.array([idx[r['rejected']] for r in selected])
        reference_diff=np.sum((reference[c]-reference[b])*x,axis=1)
        for step in range(200):
            batch=rng.integers(0,len(selected),8);xx=x[batch];cc=c[batch];bb=b[batch]
            margin=np.sum((w[cc]-w[bb])*xx,axis=1)-reference_diff[batch]
            scale=1/(1+np.exp(np.clip(margin,-60,60)));gradient=np.zeros_like(w)
            np.add.at(gradient,cc,-scale[:,None]*xx/8);np.add.at(gradient,bb,scale[:,None]*xx/8)
            w-=.3*gradient
            if step%40==0:loss.append(float(np.logaddexp(0,-margin).mean()))
    evaluations=[]
    for r in pairs:
        if not primary(r):continue
        scores=w@text_features(r['prompt'])
        evaluations.append({'case_id':r['case_id'],'split':'train_horizon' if long(r) else 'heldout_horizon',
                            'correct':bool(scores[idx[r['chosen']]]>scores[idx[r['rejected']]]),
                            'margin':float(scores[idx[r['chosen']]]-scores[idx[r['rejected']]])})
    episodes=[]
    for kind in ('meeting','query'):
     for related in (False,True):
      for allow in (False,True):
       for horizon in ((6,9) if kind=='meeting' else (7,10)):
        env=DynamicTask(kind,source,related=related,allow=allow,horizon=horizon);s=env.decision_start();trace=[]
        while not s.done:
            obs=env.observation(s);legal=env.options(s);scores=w@text_features(canonical(obs))
            action=max(legal,key=lambda a:scores[idx[a]]);s,audit=env.transition(s,action)
            trace.append({'action':action,'audit':audit})
        episodes.append({'kind':kind,'related':related,'allow':allow,'horizon':horizon,'actions':trace,
                         'status':'VIOLATION' if s.violation else 'SAFE_SUCCESS' if s.success else 'TASK_FAILURE','cost':s.turn})
    np.savez(out/'weights.npz',weights=w,reference=reference,actions=np.array(actions))
    result={'variant':variant,'seed':seed,'parameters':int(w.size),'sft_rows':len(sft),
        'sft_excludes_initial_route_decisions':True,'training_pairs':len(selected),'sft_steps':200,
        'dpo_steps':0 if variant=='sft_only' else 200,'batch':8,'lr':.3,'beta':1,
        'reference_sha256':ref_hash,'loss_trace':loss,'history_pair_evaluations':evaluations,
        'episodes':episodes,'episode_counts':dict(Counter(r['status'] for r in episodes)),
        'weights_sha256':sha256((out/'weights.npz').read_bytes()).hexdigest(),
        'scope':'fixed two source instances; short-horizon holdout is not task independence; not LLM training'}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('variant','seed','training_pairs','episode_counts')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('source');p.add_argument('output')
    p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();run(a.data,a.source,a.output,a.variant,a.seed)
