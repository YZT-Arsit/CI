"""Finite candidate enumeration using only an authorized Observation."""
from itertools import combinations
from .contracts import Action,CandidateBatch,Observation


class MultiRuleCandidateGenerator:
    version='multi_rule_enumerator_v1'
    deterministic=True

    def generate(self,obs:Observation,limit:int=12)->CandidateBatch:
        local={f.id:f for f in obs.local_facts}
        def sends(recipient,ids):
            return [Action(obs.actor,'SEND',recipient,part,' '.join(local[i].text for i in part))
                    for n in range(1,len(ids)+1) for part in combinations(sorted(ids),n)]
        actions=[]
        if obs.actor=='A':
            if obs.awaiting_notification:
                target=local['notification_target'].value
                actions=[Action('A','NOTIFY',target,answer=a) for a in ('yes','no')]
            else:
                actions=[Action('A','SKIP'),*sends('R',[k for k in local if k.startswith('context_')])]
        elif obs.actor=='B':
            actions=sends('R',[k for k in local if k.startswith(('payload_','decision_'))])
            if 'task_rule' not in local and not obs.pending_request:
                actions.append(Action('B','REQUEST_RULES','R'))
            if {'payload_0','task_rule'}.issubset(local) and not any(k.startswith('decision_') for k in local):
                actions.append(Action('B','COMPUTE_TASK'))
        elif obs.actor=='R':
            if obs.pending_request and 'task_rule' in local:
                actions+=sends('B',['task_rule'])
            if {'payload_0','task_rule'}.issubset(local) or any(k.startswith('decision_') for k in local):
                actions += [Action('R','FINALIZE',answer=a) for a in ('yes','no')]
        actions.append(Action(obs.actor,'ABSTAIN'))
        unique=list({a.key():a for a in actions}.values())
        return CandidateBatch(tuple(unique[:limit]),len(unique)<=limit,self.version)
