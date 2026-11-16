"""Candidate generation: authorized-observation-only, with explicit coverage flags."""
from __future__ import annotations
import json
from itertools import combinations
from typing import Callable,Sequence
from .contracts import Action,Observation,CandidateBatch

class ToyCandidateGenerator:
    """Exhaustive only for the finite toy domain at each observation.

    This is a testing oracle for *action enumeration*, NOT CI or task correctness.
    It never receives the full world state, labels, forbidden facts, or future outcomes.
    """
    version='toy_enumerator_v2'
    deterministic=True
    def generate(self,obs:Observation,limit:int=12)->CandidateBatch:
        local={f.id:f for f in obs.local_facts}
        def send(recipient:str,ids:tuple[str,...])->Action:
            body=' '.join(local[i].text for i in ids)
            return Action(obs.actor,'SEND',recipient,ids,body)
        def sends(recipient:str,predicates:set[str])->list[Action]:
            ids=sorted(k for k,f in local.items() if f.predicate in predicates)
            return [send(recipient,part) for n in range(1,len(ids)+1) for part in combinations(ids,n)]
        actions=[]
        if obs.actor=='A':
            if obs.awaiting_notification:
                actions=[Action('A','NOTIFY',recipient=local['identity'].value,answer='yes'),
                         Action('A','NOTIFY',recipient=local['identity'].value,answer='no'),
                         Action('A','ABSTAIN')]
            else:
                actions=[Action('A','SKIP'),*sends('R',{'identity'}),Action('A','ABSTAIN')]
        elif obs.actor=='B':
            actions.extend(sends('R',{'band','eligible','meeting_time'}))
            if not obs.pending_request and 'rule' not in local:
                actions.append(Action('B','REQUEST_RULES',recipient='R'))
            if {'band','rule'}.issubset(local) and 'eligible' not in local:
                actions.append(Action('B','COMPUTE_ELIGIBILITY'))
            actions.append(Action('B','ABSTAIN'))
        elif obs.actor=='R':
            if obs.pending_request and 'rule' in local:
                actions.append(send('B',('rule',)))
            actions.extend([Action('R','FINALIZE',answer='yes'),Action('R','FINALIZE',answer='no')])
            actions.append(Action('R','ABSTAIN'))
        unique=list({a.key():a for a in actions}.values())
        return CandidateBatch(tuple(unique[:limit]),exhaustive=len(unique)<=limit,
                              generator_version=self.version)

class JSONCandidateGenerator:
    """Model adapter with a strict interface. callback sees ONLY Observation.as_prompt().

    callback(prompt, n) returns JSON objects or strings. Never pass CI oracle,
    full WorldState, or future reference rollouts to the generator callback.
    The external model may hallucinate facts: the environment must reject them.
    """
    def __init__(self,callback:Callable[[str,int],Sequence[dict|str]],version:str='model_json_v1'):
        self.callback=callback;self.version=version
        self.deterministic=False
    def generate(self,observation:Observation,limit:int=8)->CandidateBatch:
        data=self.callback(observation.as_prompt(),limit)
        parsed=[];bad=0;seen=set()
        for item in data:
            try:
                obj=json.loads(item) if isinstance(item,str) else item
                act=Action.from_mapping(obj)
                if act.actor!=observation.actor:raise ValueError('actor mismatch')
                if act.key() not in seen:parsed.append(act);seen.add(act.key())
            except (ValueError,TypeError,KeyError,json.JSONDecodeError):
                bad+=1
        return CandidateBatch(tuple(parsed[:limit]),exhaustive=False,
                              generator_version=self.version,rejected_parse_count=bad)
