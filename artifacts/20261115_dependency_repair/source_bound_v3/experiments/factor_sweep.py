"""One bounded factor sweep per process. Synthetic mechanism checks, not LLM results."""
import argparse
from copy import deepcopy
from dataclasses import asdict,replace
import time

from kci import ContinuationEvaluator, certify_preference, DeterministicPolicy, PairedReferenceComparator
from kci.contracts import Action, CandidateBatch, Message, Verdict
from kci.rewards import outcome_from_trace
from experiments.common import world, reference, prefix, policy, rollout, write_result


def knowledge():
    rows=[]
    for index in range(8):
        for permitted in (False,True):
            env=world(index,permitted=permitted)
            for history in ('none','identity','identity_other','background','draft','duplicate'):
                st=env.initial()
                if history=='background':st.local_fact_ids['R'].add('identity')
                if history in {'identity','identity_other','duplicate'}:
                    key='identity_other' if history=='identity_other' else 'identity'
                    st,_=prefix(env,[env.act('A','SEND','R',(key,))],st)
                    if history=='duplicate':st.receipts.append(replace(st.receipts[-1],event_id='duplicate_delivery'))
                else:
                    if history=='draft':st.receipts.append(Message('draft','A','R',env.purpose,('identity',),env.rendered(('identity',)),False))
                    st,_=prefix(env,[env.act('A','SKIP')],st)
                original=repr(st.signature());after,audit=env.transition(st,env.act('B','SEND','R',('band',)))
                expected='violation' if not permitted and history in {'identity','background','duplicate'} else 'valid'
                rows.append({'world':index,'permitted':permitted,'history':history,'verdict':audit.verdict.value,
                             'expected':expected,'input_unchanged':original==repr(st.signature()),
                             'pass':audit.verdict.value==expected and original==repr(st.signature())})
    return rows


def history_length():
    rows=[]
    for count in (0,4,16,64,256):
        for identity in (False,True):
            env=world();st=env.initial()
            # Controlled state injection for sensitivity testing only, not a replay-certified training episode.
            st.receipts.extend(Message(f'context_{j}','B','R',env.purpose,('time',),env.rendered(('time',))) for j in range(count))
            first=env.act('A','SEND','R',('identity',)) if identity else env.act('A','SKIP')
            st,_=prefix(env,[first],st)
            t=time.perf_counter()
            for _ in range(40):_,audit=env.transition(st,env.act('B','SEND','R',('band',)))
            rows.append({'history_length':count,'identity':identity,'verdict':audit.verdict.value,
                         'transition_ms':1000*(time.perf_counter()-t)/40,
                         'prompt_chars':len(env.observation(st,'B').as_prompt()),
                         'history_provenance':'controlled_state_injection_not_prior_episode_replay',
                         'pass':audit.verdict==(Verdict.VIOLATION if identity else Verdict.VALID)})
    return rows


def search_budget():
    rows=[]
    for nodes in (1,8,32,128,1024,3000):
        for actions in (1,2,4,12):
            env=world();cce=ContinuationEvaluator(env,reference(env,nodes,actions));st=env.initial()
            a=cce.evaluate(st,env.act('A','SKIP'));b=cce.evaluate(st,env.act('A','SEND','R',('identity',)))
            pair=certify_preference(a,b)
            statuses=[a.reference.status.value,b.reference.status.value]
            rows.append({'nodes':nodes,'action_limit':actions,'statuses':statuses,
                         'costs':[a.reference.best_success_cost,b.reference.best_success_cost],
                         'expanded':[a.reference.expanded_states,b.reference.expanded_states],
                         'preference':None if pair is None else pair.source,
                         'pass':pair is None or 'unknown' not in statuses})
    return rows


def horizons_costs():
    rows=[]
    for horizon in (3,4,6,7,9,12):
        for costs in ({},{'SEND':3.0,'COMPUTE_ELIGIBILITY':0.25},{'SKIP':5.0}):
            env=world(horizon=horizon,costs=costs);cce=ContinuationEvaluator(env,reference(env));st=env.initial()
            a=cce.evaluate(st,env.act('A','SKIP'));b=cce.evaluate(st,env.act('A','SEND','R',('identity',)))
            pair=certify_preference(a,b)
            cost=lambda kind:costs.get(kind,1.0)
            expected_a=sum(cost(k) for k in ('SKIP','SEND','FINALIZE','NOTIFY')) if horizon>=4 else None
            expected_b=sum(cost(k) for k in ('SEND','REQUEST_RULES','SEND','COMPUTE_ELIGIBILITY','SEND','FINALIZE','NOTIFY')) if horizon>=7 else None
            expected_choice=('SKIP' if expected_b is None or expected_a<expected_b else 'SEND') if expected_a is not None else None
            rows.append({'horizon':horizon,'action_costs':costs,'statuses':[a.reference.status.value,b.reference.status.value],
                         'costs':[a.reference.best_success_cost,b.reference.best_success_cost],
                         'chosen':None if pair is None else pair.chosen.kind,
                         'expected_costs':[expected_a,expected_b],'expected_choice':expected_choice,
                         'pass':a.reference.best_success_cost==expected_a and b.reference.best_success_cost==expected_b
                         and (None if pair is None else pair.chosen.kind)==expected_choice})
    return rows


def reference_policies():
    rows=[]
    for permitted in (False,True):
        for initial in ('SKIP','SEND'):
            for mode in ('aware','blind','conservative','abstain','wrong_answer'):
                env=world(permitted=permitted);trace=rollout(env,policy(env,mode,initial));out=outcome_from_trace(trace)
                expected_violation=mode=='blind' and initial=='SEND' and not permitted
                expected_safe_success=not expected_violation and mode not in {'abstain','wrong_answer'}
                rows.append({'permitted':permitted,'initial':initial,'policy':mode,**asdict(out),
                             'safe_success':out.safe_success,'actions':trace.actions(),
                             'evidence':'handwritten_reference_control','expected_safe_success':expected_safe_success,
                             'pass':out.safe_success==expected_safe_success and out.ci_compliant==(not expected_violation)})
    return rows


def statistics():
    rows=[]
    for n in (16,64,256):
        for mode in ('aware','blind','conservative','abstain'):
            env=world();comp=PairedReferenceComparator(env,DeterministicPolicy(policy(env,mode),'control_'+mode),n=n)
            r=comp.compare(env.initial(),env.act('A','SKIP'),env.act('A','SEND','R',('identity',)))
            rows.append({'n':n,'mode':mode,'status':r.status,'ci_diff':r.ci_diff_mean,'safe_success_diff':r.success_diff_mean,
                         'radius':r.ci_diff_radius,'cost_diff':r.conditional_cost_diff_mean,'reason':r.reason,
                         'certified':r.preference is not None,
                         'pass':r.preference is None if mode in {'aware','conservative','abstain'} else r.preference is not None})
    return rows


def semantic_faults():
    rows=[];env=world();st,_=prefix(env,[env.act('A','SKIP')]);good=env.act('B','SEND','R',('band',))
    cases=[('canonical',good,Verdict.VALID),('extra_secret',replace(good,body=good.body+' A private fact.'),Verdict.UNKNOWN),
           ('paraphrase',replace(good,body='The salary category for this identifier is B1.'),Verdict.UNKNOWN),
           ('unknown_recipient',replace(good,recipient='nobody'),Verdict.INVALID_ACTION),
           ('hidden_fact',replace(good,fact_ids=('identity',)),Verdict.INVALID_ACTION),
           ('wrong_actor',replace(good,actor='A'),Verdict.INVALID_ACTION),
           ('duplicate_fact',replace(good,fact_ids=('band','band')),Verdict.INVALID_ACTION)]
    for label,action,expected in cases:
        before=repr(st.signature());after,aud=env.transition(st,action)
        rows.append({'case':label,'verdict':aud.verdict.value,'expected':expected.value,
                     'pass':aud.verdict==expected and before==repr(st.signature())})
    return rows


SUITES={f.__name__:f for f in (knowledge,history_length,search_budget,horizons_costs,reference_policies,statistics,semantic_faults)}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=SUITES,required=True);p.add_argument('--output',required=True)
    args=p.parse_args();t=time.perf_counter();rows=SUITES[args.suite]()
    result={'suite':args.suite,'evidence_class':'synthetic_mechanism_only','seconds':time.perf_counter()-t,
            'cases':len(rows),'passed':sum(r['pass'] for r in rows),'rows':rows}
    write_result(args.output,result);print({k:v for k,v in result.items() if k!='rows'})
    raise SystemExit(0 if result['passed']==result['cases'] else 1)
