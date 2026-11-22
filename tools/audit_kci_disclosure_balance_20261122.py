"""Paired CPU diagnostic for KCI reward/data balance at source-release.

Contrast negative-only downstream CI repair vs reward-aligned positive
legal-disclosure pairs without modifying the native worlds.
All source actions are grounded on actual same-state tool transitions.
This is a small diagnostic of over-conservatism, not LLM training.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
from copy import deepcopy
import sys,json,random,statistics
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import compare_kci_chain_methods_cpu_action_interaction as T
from chain_shared_actor_cpu_action_interaction import score_actions
from kci_exact_policy_continuation_cpu import environment_public,enumerate_decisions,FrozenPolicyValue
from run_kci_role_local_credit_cpu import load_base
from run_kci_downstream_residual_cpu import RepairedDownstream
from run_kci_cpu_policy_conditioned_gate import greedy_eval
from run_kci_native_repair_mc_bridge_cpu import immediate_downstream_repairs
torch.set_num_threads(2)
OUT=ROOT/'artifacts'/'20261122_kci_safe_disclosure_balance_CPU_v1'
SEEDS=(7,1234,2025)
STEPS=80
LR=.03

def safe_raw_pairs(base):
    points=[]
    count=Counter()
    for row in T.GROUP['train']:
        env=environment_public(row)
        positions,_=enumerate_decisions(env)
        val=FrozenPolicyValue(env,base)
        for p in positions:
            if p['phase']!='release':continue
            if set(p['verdicts'].values()) != {'VALID'}:continue
            count['both_immediately_legal']+=1
            q_raw=val.q(p['state'],'SEND_RAW_R')
            q_verify=val.q(p['state'],'VERIFY_PRIVATE_R')
            # reward cost benefit, no additional CI risk, no less successful.
            if (q_raw[0]>q_verify[0]+.005 and
                q_raw[2]<=q_verify[2]+1e-4 and
                q_raw[1]>=q_verify[1]-1e-4):
                points.append(T.point(p['obs'],'SEND_RAW_R',
                                      rejected='VERIFY_PRIVATE_R',
                                      weight=1.,origin='actual_legal_raw_cheaper_and_task_sufficient'))
            else:count['both_valid_but_raw_not_dominant']+=1
    count['safe_raw_positive_pairs']=len(points)
    return points,dict(count)

def raw_prob(model,rows):
    data=Counter()
    probs={'safe':[],'unsafe':[]}
    for row in rows:
        env=environment_public(row);states,_=enumerate_decisions(env)
        for p in states:
            if p['phase']!='release':continue
            options=list(p['actions'])
            with torch.no_grad():
                logits=score_actions(model,p['obs'],options)
                prob=torch.softmax(logits,dim=0).cpu().tolist()[options.index('SEND_RAW_R')]
            tag='safe' if p['verdicts']['SEND_RAW_R']=='VALID' else 'unsafe'
            probs[tag].append(prob)
    return {k:{'count':len(v),'mean_p_raw':statistics.mean(v) if v else None} for k,v in probs.items()}

def experiment(seed,negative):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    base,source=load_base(seed)
    positive,diagnosis=safe_raw_pairs(base)
    r={}
    for mode in ('negative_only','balanced_CI_and_legitimate'):
        net=RepairedDownstream(base)
        for p in net.B.parameters():p.requires_grad_(False)
        neg_cache=T.feature_cache(negative)
        neg_ref=T.ref_diff(net,neg_cache)
        if mode=='balanced_CI_and_legitimate':
            pos_cache=T.feature_cache(positive)
            pos_ref=T.ref_diff(net,pos_cache)
        opt=torch.optim.Adam(net.R.parameters(),lr=LR)
        for step in range(STEPS):
            opt.zero_grad()
            loss_n=T.supervised_loss(net,neg_cache,'dpo',neg_ref)
            if mode=='balanced_CI_and_legitimate':
                loss_p=T.supervised_loss(net,pos_cache,'dpo',pos_ref)
                loss=.5*loss_n+.5*loss_p
            else:loss=loss_n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.R.parameters(),3)
            opt.step()
        ev={split:{k:v for k,v in greedy_eval(net,rows).items()
                    if k!='per_world'} for split,rows in T.GROUP.items() if split!='train'}
        prob={split:raw_prob(net,rows) for split,rows in T.GROUP.items()}
        r[mode]={
          'train_positive_pairs':len(positive) if mode=='balanced_CI_and_legitimate' else 0,
          'train_immediate_violation_pairs':len(negative),
          'train_label_audit':diagnosis,
          'eval':ev,
          'raw_prob':prob,
        }
        test=ev['exploratory_test']
        print('RESULT',seed,mode,'soft_return',round(test['mean_policy_expected_return'],6),
              'soft_violation',round(test['mean_policy_expected_CI_violation'],6),
              'legal_raw_prob',round(prob['exploratory_test']['safe']['mean_p_raw'],5),
              'illegal_raw_prob',round(prob['exploratory_test']['unsafe']['mean_p_raw'],5),
              'greedy_safe',test['greedy_safe_success'],flush=True)
    return {'seed':seed,'runs':r}

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    neg,c=immediate_downstream_repairs()
    results=[experiment(seed,neg) for seed in SEEDS]
    aggregate={}
    for mode in ('negative_only','balanced_CI_and_legitimate'):
        test=[r['runs'][mode]['eval']['exploratory_test'] for r in results]
        aggregate[mode]={
          'soft_expected_return':[x['mean_policy_expected_return'] for x in test],
          'soft_CI_violation_probability':[x['mean_policy_expected_CI_violation'] for x in test],
          'greedy_safe_success':[x['greedy_safe_success'] for x in test],
          'legal_raw_probability':[r['runs'][mode]['raw_prob']['exploratory_test']['safe']['mean_p_raw'] for r in results],
          'illegal_raw_probability':[r['runs'][mode]['raw_prob']['exploratory_test']['unsafe']['mean_p_raw'] for r in results],
        }
    report={'status':'TEST_LEGAL_DISCLOSURE_POSITIVE_VS_VIOLATION_REPAIR_DPO',
            'all_cases_existing_native_no_data_modification':True,
            'train_negative_from_actual_immediate_verdict':True,
            'positive_raw_from_accepted_native_reference_Qpi_audit_only':True,
            'models':'frozen shared Horn actor; R-role 6-weight residual DPO',
            'no_LLM':True,
            'test_seen_by_developers':True,
            'negative_counts':c,
            'paired_seeds':SEEDS,'summary':aggregate,'runs':results}
    (OUT/'RESULTS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('SUMMARY',json.dumps(aggregate,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
