"""Standard DPO on an explicit finite linear policy: CPU pipeline diagnostic only.

Features are derived solely from the supplied structured Observation prompt and
candidate Action. This is not a language model, no independent holdout exists,
and in-sample fitting/rollouts must not be interpreted as KCI method efficacy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time
import numpy as np

from kci import ToyCandidateGenerator
from kci.rewards import outcome_from_trace
from experiments.common import world, reference, prefix, rollout, write_result

ACTION_KEYS=['SKIP','SEND:identity','SEND:identity_other','SEND:identity,identity_other',
             'SEND:band','SEND:time','SEND:band,time','SEND:eligible','SEND:band,eligible',
             'SEND:eligible,time','SEND:band,eligible,time','SEND:rule','REQUEST_RULES',
             'COMPUTE_ELIGIBILITY','FINALIZE:yes','FINALIZE:no','NOTIFY:yes','NOTIFY:no','ABSTAIN']


def features(prompt,action,representation='structured'):
    obs=json.loads(prompt) if isinstance(prompt,str) else prompt
    act=json.loads(action) if isinstance(action,str) else action
    norms=obs['visible_ci_norms'];receipts=obs['authorized_delivered_messages']
    task_target=re.search(r'eligibility for (\w+)',obs['task'])
    target=task_target.group(1) if task_target else ''
    relevant=any(m['delivered'] and m['recipient']=='R' and target+' corresponds to ' in m['body'] for m in receipts)
    unrelated=any(m['delivered'] and m['recipient']=='R' and ' corresponds to ' in m['body'] and target+' corresponds to ' not in m['body'] for m in receipts)
    restricted=any('R must not learn a specific person-to-salary-band' in n for n in norms)
    if representation=='no_history':relevant=unrelated=False
    if representation=='no_norm':restricted=False
    local={f['id'] for f in obs['local_facts']}
    state=np.array([1,obs['role']=='A',obs['role']=='B',obs['role']=='R',restricted,relevant,unrelated,
                    'rule' in local,'eligible' in local,obs['turn']/9,obs['remaining_steps']/9,
                    obs['pending_request'],obs['awaiting_notification'],
                    any("'REQUEST_RULES': 2.0" in n for n in norms)],dtype=float)
    key=act['kind']
    if key=='SEND':key+=':'+','.join(sorted(act['fact_ids']))
    if key in {'FINALIZE','NOTIFY'}:key+=':'+act['answer']
    onehot=np.zeros(len(ACTION_KEYS)+1);onehot[ACTION_KEYS.index(key) if key in ACTION_KEYS else -1]=1
    return np.kron(onehot,state)


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-60,60)))


def metrics(theta,reference,x,rows):
    margin=x@theta
    return {'overall':float(np.mean(margin>1e-9)),
            'by_source':{s:float(np.mean(margin[[s in r['metadata']['training_sources'] for r in rows]]>1e-9)) for s in ('ci','future','repair')},
            'ties':int(np.sum(np.abs(margin)<=1e-9)),
            'mean_margin':float(np.mean(margin)),
            'reference_shift_l2':float(np.linalg.norm(theta-reference))}


def trajectory_metrics(theta,representation):
    outcomes=[];gen=ToyCandidateGenerator()
    for permitted in (False,True):
        for horizon in (6,9):
            for initial in ('start','identity','identity_other'):
                env=world(permitted=permitted,horizon=horizon)
                st=env.initial()
                if initial!='start':st,_=prefix(env,[env.act('A','SEND','R',(initial,))])
                optimal=reference(env).search(st)
                def choose(obs):
                    acts=gen.generate(obs,limit=12).actions
                    scores=[features(obs.as_prompt(),a.__dict__,representation)@theta for a in acts]
                    return acts[int(np.argmax(scores))]
                trace=rollout(env,choose,st);out=outcome_from_trace(trace)
                outcomes.append({'permitted':permitted,'horizon':horizon,'initial':initial,
                    'safe_success':out.safe_success,'ci_compliant':out.ci_compliant,'cost':out.execution_cost,
                    'actions':trace.actions(),'start_reference_status':optimal.status.value,
                    'feasible':optimal.best_trace is not None,'optimal_cost':optimal.best_success_cost})
    return {'episodes':len(outcomes),'safe_successes':sum(r['safe_success'] is True for r in outcomes),
            'confirmed_violations':sum(r['ci_compliant'] is False for r in outcomes),
            'feasible_episodes':sum(r['feasible'] for r in outcomes),
            'rows':outcomes,'scope':'same_base_world_in_sample_menu_policy_diagnostic'}


def sft_initialize(theta,rng,representation):
    """Verified synthetic trajectory SFT; same-base-world initialization diagnostic."""
    examples={};gen=ToyCandidateGenerator()
    for permitted in (False,True):
        for horizon in (6,9):
            env=world(permitted=permitted,horizon=horizon)
            for first in (None,'identity','identity_other'):
                st=env.initial()
                if first:st,_=prefix(env,[env.act('A','SEND','R',(first,))])
                result=reference(env).search(st)
                if result.best_trace is None:continue
                for step in result.best_trace.steps:
                    prompt=env.observation(step.before,step.action.actor).as_prompt()
                    actions=list(gen.generate(env.observation(step.before,step.action.actor),limit=12).actions)
                    keys=[a.key() for a in actions]
                    if step.action.key() not in keys:raise ValueError('witness action missing from menu')
                    examples[(prompt,step.action.key())]=(np.stack([features(prompt,a.__dict__,representation) for a in actions]),keys.index(step.action.key()))
    values=list(examples.values())
    for _ in range(400):
        gradient=np.zeros_like(theta)
        for i in rng.integers(0,len(values),size=8):
            design,label=values[i];scores=design@theta;scores-=np.max(scores)
            probs=np.exp(scores);probs/=probs.sum();gradient+=probs@design-design[label]
        theta-=.1*gradient/8
    return theta,len(values)


def run(rows,variant,seed):
    rng=np.random.default_rng(seed)
    rep='no_history' if 'no_history' in variant else 'no_norm' if 'no_norm' in variant else 'structured'
    x=np.stack([features(r['prompt'],r['chosen'],rep)-features(r['prompt'],r['rejected'],rep) for r in rows])
    sources=('ci',) if variant.startswith('ci_only') else ('ci','future') if variant.startswith('ci_future') else ('ci','future','repair')
    indices=np.array([i for i,r in enumerate(rows) if set(sources)&set(r['metadata']['training_sources'])])
    weights=np.array([10 if variant=='repair_x10' and 'repair' in rows[i]['metadata']['training_sources'] else 1 for i in indices],float)
    ci_weight=14 if variant=='all_sft_ci_x14' else 4 if variant=='all_sft_ci_x4' else 1
    if ci_weight>1:
        weights=np.array([ci_weight if 'ci' in rows[i]['metadata']['training_sources'] else 1 for i in indices],float)
    beta=0.1 if variant=='beta_0.1' else 10.0 if variant=='beta_10' else 1.0
    theta=rng.normal(0,.001,x.shape[1]);sft_examples=0
    if 'sft' in variant:theta,sft_examples=sft_initialize(theta,rng,rep)
    frozen=theta.copy();signs=np.ones(len(rows))
    noise=0.1 if variant=='noise_10' else 0.3 if variant=='noise_30' else 0
    flipped=rng.choice(indices,size=round(noise*len(indices)),replace=False)
    signs[flipped]=-1
    before=metrics(theta,frozen,x,rows);losses=[];checkpoints=[];used=[]
    dpo_steps=0 if variant=='sft_only' else 240
    order=None;fixed_batches={};cursors={'ci':0,'future':0}
    if variant in {'future_then_ci','ci_then_future','balanced_ci_future'}:
        # Identical within-source minibatch multisets across order variants.
        for src in ('ci','future'):
            pool=np.array([i for i,r in enumerate(rows) if src in r['metadata']['training_sources']])
            fixed_batches[src]=rng.choice(pool,size=(120,8),replace=True)
        order=['future']*120+['ci']*120
        if variant=='ci_then_future':order.reverse()
        if variant=='balanced_ci_future':rng.shuffle(order)
    for step in range(dpo_steps):
        active=indices;prob=weights/weights.sum()
        if order is not None:
            src=order[step]
            batch=fixed_batches[src][cursors[src]];cursors[src]+=1
        else:batch=rng.choice(active,size=8,replace=True,p=prob)
        used.extend(int(i) for i in batch)
        design=x[batch]*signs[batch,None]
        z=beta*(design@(theta-frozen))
        grad=np.mean((beta*(sigmoid(z)-1))[:,None]*design,axis=0)
        theta-=.1*grad
        if step%40==0:losses.append({'step':step,'batch_loss':float(np.mean(np.logaddexp(0,-z)))})
        if step==119:checkpoints.append({'step':120,'metrics':metrics(theta,frozen,x,rows)})
    # Finite-difference check of the same-prompt categorical DPO gradient.
    design=x[indices[:4]];test=rng.normal(0,.01,x.shape[1]);z=beta*(design@(test-frozen))
    analytic=np.mean((beta*(sigmoid(z)-1))[:,None]*design,axis=0)
    coordinate=int(np.argmax(np.abs(analytic)));eps=1e-6;plus=test.copy();minus=test.copy();plus[coordinate]+=eps;minus[coordinate]-=eps
    loss=lambda t:np.mean(np.logaddexp(0,-beta*(design@(t-frozen))))
    finite=(loss(plus)-loss(minus))/(2*eps)
    after=metrics(theta,frozen,x,rows)
    return {'variant':variant,'seed':seed,'representation':rep,'beta':beta,'steps':dpo_steps,'batch_size':8,
            'sft_examples':sft_examples,'sft_steps':400 if sft_examples else 0,
            'ci_sampling_weight':ci_weight,
            'learning_rate':.1,'eligible_training_pool_patterns':len(indices),
            'unique_training_patterns':len(set(used)),'evaluation_patterns':len(rows),
            'fixed_source_batches':{s:b.tolist() for s,b in fixed_batches.items()},
            'flipped_pattern_ids':[rows[i]['metadata']['canonical_pattern_id'] for i in flipped],
            'before':before,'after':after,'loss_trace':losses,
            'intermediate_checkpoints':checkpoints,'source_schedule':order,
            'gradient_check_absolute_error':float(abs(finite-analytic[coordinate])),
            'gradient_check_pass':bool(abs(finite-analytic[coordinate])<1e-6),
            'trajectory_diagnostic':trajectory_metrics(theta,rep),
            'parameters':theta.tolist(),'frozen_reference_parameters':frozen.tolist(),
            'evidence_class':'finite_linear_DPO_proxy_NOT_LLM_training',
            'evaluation_scope':'same_corpus_no_independent_holdout_NO_generalization_claim'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);args=p.parse_args()
    rows=[json.loads(x) for x in Path(args.dataset).read_text(encoding='utf-8').splitlines()]
    t=time.perf_counter();result=run(rows,args.variant,args.seed);result['seconds']=time.perf_counter()-t
    write_result(args.output,result);print({k:result[k] for k in ('variant','seed','seconds','gradient_check_pass','after')})
    raise SystemExit(0 if result['gradient_check_pass'] else 1)
