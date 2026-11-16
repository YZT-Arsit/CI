"""Frozen observation-only reference control, not the trained research method."""
import random
from .dynamic_tasks import canonical


class DynamicReference:
    version='observation_reference_v1'
    def __init__(self,mode='safe'):
        if mode not in ('safe','careless','mixed'):raise ValueError('unknown reference')
        self.mode=mode

    def choose(self,observation,seed):
        options=observation['actions'];phase=observation['phase'];kind=observation['task']
        if len(options)==1:return options[0]
        # This relation check uses only authorized receipts and routing metadata.
        messages=[m['content'] for m in observation['authorized_receipts'] if m['recipient']=='R']
        permitted=observation['derived_relation_permission']=='allowed'
        if kind=='meeting':
            meta=observation['authorized_routing_metadata'];link=meta['upcoming_case_link']
            related=any(x['type']=='case_cause' and x['case']==link['case'] for x in messages)
            roster=any(x['type']=='roster' and x['alias']==link['alias'] for x in messages)
            risk=related and not permitted
            if phase=='choose':return 'PRIVATE_BOOKING' if risk else 'SEND_ROSTER'
            if phase=='hr':
                safe='REQUEST_SANITIZED' if risk and roster else 'SEND_CASE_LINK'
                raw='SEND_CASE_LINK'
            else:return next(x for x in options if x!='ABORT')
        else:
            meta=observation['authorized_routing_metadata']
            names={r['company_id'] for x in messages if x['type']=='owner_names' for r in x['rows']}
            ids=set(meta['city_bundle_schema']['owner_ids'])
            related=any(r['company_id'] in names and r['owner_id'] in ids for r in meta['owner_index'])
            indexed=any(x['type']=='owner_index' for x in messages)
            risk=related and not permitted
            if phase=='choose':return 'OPEN_VAULT' if risk else 'OPEN_INDEX'
            if phase=='collect':
                if 'FETCH_COUNTS' in options:return 'FETCH_COUNTS'
                safe='REQUEST_CITY_PROJECTION' if risk and indexed else 'FETCH_CITY_BUNDLE'
                raw='FETCH_CITY_BUNDLE'
            else:return next(x for x in options if x!='ABORT')
        if self.mode=='careless':return raw
        if self.mode=='mixed' and random.Random(seed).randrange(2):return raw
        return safe


def rollout(env,state,policy,seed):
    s=state;trace=[]
    while not s.done:
        observation=env.observation(s);action=policy.choose(observation,seed)
        n,audit=env.transition(s,action)
        trace.append({'observation':observation,'action':action,'audit':audit,'after_turn':n.turn})
        if audit['verdict']=='UNKNOWN':return n,trace
        s=n
    return s,trace
