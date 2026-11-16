import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json
from pathlib import Path
from hashlib import sha256
from collections import defaultdict,Counter
from dataclasses import asdict
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from experiments.text_policy_probe import text_features
from kci.dynamic_tasks import DynamicTask,canonical
SOURCE=ROOT/'artifacts/20261113_iteration2/source_anchors'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines()]
def save(n,x):(RUN/n).write_text(json.dumps(x,indent=2),encoding='utf-8')

def finite():
    references=defaultdict(set);episode_checks=0;summary={};gradient_errors=[]
    for path in sorted((RUN/'finite_v2').glob('*/result.json')):
        result=read(path);weightfile=path.with_name('weights.npz');assert sha256(weightfile.read_bytes()).hexdigest()==result['weights_sha256']
        arrays=np.load(weightfile);w=arrays['weights'];reference=arrays['reference'];actions=list(arrays['actions']);idx={a:i for i,a in enumerate(actions)}
        h=sha256(reference.tobytes()).hexdigest();assert h==result['reference_sha256'];references[result['seed']].add(h)
        for episode in result['episodes']:
            e=DynamicTask(episode['kind'],SOURCE,related=episode['related'],allow=episode['allow'],horizon=episode['horizon']);s=e.decision_start()
            for step in episode['actions']:
                scores=w@text_features(canonical(e.observation(s)))
                action=max(e.options(s),key=lambda a:scores[idx[a]])
                assert action==step['action'];s,audit=e.transition(s,action);assert canonical(audit)==canonical(step['audit'])
            status='VIOLATION' if s.violation else 'SAFE_SUCCESS' if s.success else 'TASK_FAILURE'
            assert status==episode['status'] and s.turn==episode['cost'];episode_checks+=1
        group=summary.setdefault(result['variant'],{'episodes':Counter(),'pair_correct':Counter(),'pair_total':Counter(),'models':0})
        group['models']+=1;group['episodes'].update(result['episode_counts'])
        for p in result['history_pair_evaluations']:
            group['pair_total'][p['split']]+=1;group['pair_correct'][p['split']]+=p['correct']
        # Independent scalar finite differences for the implemented linear DPO rule.
        rng=np.random.default_rng(result['seed']);x=rng.normal(size=(4,w.shape[1]));c=np.array([0,0,1,2]);b=np.array([2,1,2,0])
        refdiff=np.sum((reference[c]-reference[b])*x,axis=1)
        def loss(v):return np.logaddexp(0,-(np.sum((v[c]-v[b])*x,axis=1)-refdiff)).mean()
        margin=np.sum((w[c]-w[b])*x,axis=1)-refdiff;coef=-1/(1+np.exp(margin));gradient=np.zeros_like(w)
        np.add.at(gradient,c,coef[:,None]*x/4);np.add.at(gradient,b,-coef[:,None]*x/4)
        for ij in [(0,0),(1,1),(2,2)]:
            plus=w.copy();minus=w.copy();plus[ij]+=1e-5;minus[ij]-=1e-5
            gradient_errors.append(abs((loss(plus)-loss(minus))/2e-5-gradient[ij]))
    assert all(len(v)==1 for v in references.values()) and max(gradient_errors)<1e-7
    save('finite_audit.json',{'episode_replays':episode_checks,'reference_equal_by_seed':True,
                            'gradient_max_error':max(gradient_errors),'arms':summary,'all_passed':True})

def qwen():
    root=RUN/'qwen_v2';results=read(root/'results.json');stats=Counter();pairs=defaultdict(list);tokens=Counter();times=[]
    for i,row in enumerate(results):
        case=row['case'];e=DynamicTask(case['kind'],SOURCE,related=case['related'],allow=case['allow'],horizon=case['horizon']);s=e.decision_start()
        for a in row['prefix'][1:]:s,audit=e.transition(s,a);assert audit['verdict']=='VALID'
        request=read(root/f'call_{i:02d}/request.json');receipt=read(root/f'call_{i:02d}/worker_receipt.json')
        assert e.observation(s)==request['authorized_observation'];assert row['menu']==request['menu']
        times.append(receipt['worker_seconds']);response=receipt['response']
        if response and response['status']=='completed':
            st=response['final_statistics'];tokens['prompt']+=st.get('prompt_eval_count',0);tokens['generated']+=st.get('eval_count',0);tokens['completed']+=1
        if row['action']:
            assert row['menu'][int(response['response_text'].strip())]==row['action']
            n,audit=e.transition(s,row['action']);assert canonical(audit)==canonical(row['audit'])
            for step in row['hybrid_suffix']['trace']:
                assert e.observation(n)==step['observation'];n,a=e.transition(n,step['action']);assert canonical(a)==canonical(step['audit'])
            assert n.success==row['hybrid_suffix']['success']
        stage=case['stage'];stats[stage+'_correct']+=row.get('matches_exact_preference',False);stats[stage+'_count']+=1
        if stage=='initial' and not case['allow']:pairs[(case['kind'],case['horizon'])].append(row)
    stats['joint_history_groups']=len(pairs)
    stats['joint_history_correct']=sum(len(v)==2 and all(x.get('matches_exact_preference',False) for x in v) for v in pairs.values())
    stats['offline_repairable_decisions']=sum('offline_repair_action' in r and r['offline_repair_action'] is not None for r in results)
    assert read(root/'model_release.json')['target_model_absent_after_release']
    save('qwen_audit.json',{'replayed_decisions':len(results),'metrics':dict(stats),'completed_tokens':dict(tokens),
          'max_request_seconds':max(times),'total_request_seconds':sum(times),'model_released':True,
          'actual_full_qwen_episodes':0,'all_passed':True})

def data():
    training=rows(RUN/'training_v3/all_preferences.jsonl');encoded={}
    for p in (RUN/'sequence_v3/encoded_chunks').glob('*.jsonl'):
        for x in rows(p):encoded[x['provenance']['record_id']]=x
    primary={x['metadata']['case_id']:x for x in training if x['metadata']['history_flip_primary']}
    checks=[]
    for group in rows(RUN/'dynamic_data_v3/history_pairs.jsonl'):
        a,b=[encoded[primary[c]['metadata']['record_id']] for c in group['cases']]
        assert len(a['prompt_ids'])==len(b['prompt_ids'])
        checks.append({'cases':group['cases'],'prompt_tokens':len(a['prompt_ids'])})
    cases={x['case_id']:x['contract'] for x in read(RUN/'dynamic_data_v3/summary.json')['cases']}
    repairs=rows(RUN/'dynamic_data_v3/repairs.jsonl');unique_states=set();privacy=0
    for r in repairs:
        c=cases[r['case_id']];e=DynamicTask(c['kind'],SOURCE,related=c['related'],allow=c['allow'],horizon=c['horizon']);s=e.decision_start()
        for action in r['prefix'][1:]:s,a=e.transition(s,action);assert a['verdict']=='VALID'
        bad,a=e.transition(s,r['failed_action']);privacy+=a['verdict']=='VIOLATION'
        assert a['verdict']=='VIOLATION' or e.exact(bad)['min_total_cost'] is None
        good,a=e.transition(s,r['repair_action']);assert a['verdict']=='VALID'
        for action in r['witness']:good,a=e.transition(good,action);assert a['verdict']=='VALID'
        assert good.success and not good.violation;unique_states.add((r['case_id'],r['state_id']))
    save('data_replay_audit.json',{'token_equal_history_groups':checks,'encoded_rows':len(encoded),
         'repair_diagnostics':len(repairs),'distinct_repair_states':len(unique_states),'immediate_CI_repairs':privacy,
         'repair_only_pairs':read(RUN/'dynamic_data_v3/summary.json')['repair_pairs_not_in_main'],'all_passed':True})

if __name__=='__main__':{'finite':finite,'qwen':qwen,'data':data}[sys.argv[1]]()
