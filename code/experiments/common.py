from dataclasses import replace, asdict
from pathlib import Path
import json
import time

from kci import ToyCIEnv, ToyCandidateGenerator, ReferenceSearch, SearchBudget
from kci.contracts import Fact, Message, Trace, TraceStep, Verdict
from kci.toy_env import FACTS


def world(index=0, *, permitted=False, horizon=9, costs=None):
    """Original synthetic worlds; one fixed relation family, no external data."""
    entity=f'K{1000+index}'; other=f'K{2000+index}'; person=f'Person{index:03d}'
    band=('B1','B2','B3')[index % 3]
    facts=dict(FACTS)
    facts.update({
        'identity':Fact('identity','identity',entity,person,f'{entity} corresponds to {person}.'),
        'identity_other':Fact('identity_other','identity',other,person,f'{other} corresponds to {person}.'),
        'band':Fact('band','band',entity,band,f'{entity} has salary band {band}.'),
        'eligible':Fact('eligible','eligible',entity,'yes',f'{entity} is eligible for the project.'),
        'time':Fact('time','meeting_time',entity,'14:00',f'{entity} meeting time is 14:00.'),
        'person_band':Fact('person_band','person_band',person,band,f'{person} has salary band {band}.'),
    })
    return ToyCIEnv(horizon,facts=facts,world_id=f'synthetic_world_{index:03d}',
                    forbidden_inferences=set() if permitted else None,action_costs=costs)


def reference(env, nodes=3000, actions=12, generator=None):
    return ReferenceSearch(env,generator or ToyCandidateGenerator(),SearchBudget(nodes,actions))


def prefix(env, actions, state=None):
    st=env.initial() if state is None else state
    steps=[]
    for action in actions:
        after,audit=env.transition(st,action)
        steps.append(TraceStep(st,action,audit,after));st=after
        if audit.verdict!=Verdict.VALID or st.done:break
    return st,Trace(tuple(steps))


def policy(env, mode='aware', initial='SKIP'):
    """Handwritten reference control, not a learned model; uses observation only."""
    def choose(obs):
        local={f.id:f for f in obs.local_facts}
        if mode=='abstain':return env.act(obs.actor,'ABSTAIN')
        if obs.actor=='A':
            if obs.awaiting_notification:
                decisions=[m.body for m in obs.authorized_receipts if m.recipient=='A']
                answer='no' if decisions and decisions[-1].endswith(': no.') else 'yes'
                return env.act('A','NOTIFY',local['identity'].value,answer=answer)
            return env.act('A','SKIP') if initial=='SKIP' else env.act('A','SEND','R',('identity',))
        if obs.actor=='R':
            if obs.pending_request:return env.act('R','SEND','B',('rule',))
            return env.act('R','FINALIZE',answer='no' if mode=='wrong_answer' else 'yes')
        if 'eligible' in local:return env.act('B','SEND','R',('eligible',))
        if 'rule' in local:return env.act('B','COMPUTE_ELIGIBILITY')
        forbidden=any('R must not learn a specific person-to-salary-band' in n for n in obs.visible_ci_norms)
        entity=local['band'].subject
        identity_seen=any(m.recipient=='R' and m.delivered and
                          f'{entity} corresponds to ' in m.body for m in obs.authorized_receipts)
        if mode=='conservative' or (mode!='blind' and forbidden and identity_seen):
            return env.act('B','REQUEST_RULES','R')
        return env.act('B','SEND','R',('band',))
    return choose


def rollout(env, choose, state=None):
    st=env.initial() if state is None else state; steps=[]
    while not st.done and st.turn<st.max_turns:
        action=choose(env.observation(st,st.actor))
        after,audit=env.transition(st,action)
        steps.append(TraceStep(st,action,audit,after));st=after
        if audit.verdict!=Verdict.VALID:break
    return Trace(tuple(steps))


def write_result(path, data):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def trace_evidence(trace):
    if trace is None:return None
    return [{'action':asdict(s.action),'audit':asdict(s.audit),'before':repr(s.before.signature()),
             'after':repr(s.after.signature()),'done':s.after.done,'success':s.after.success,
             'cost':s.after.cost} for s in trace.steps]
