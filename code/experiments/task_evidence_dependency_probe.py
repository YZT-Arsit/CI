"""Common Task Evidence Dependency Test for both repaired workflows."""
from copy import deepcopy
from dataclasses import asdict
import argparse,json
from pathlib import Path
from kci.dynamic_tasks import DynamicTask

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/20261113_iteration2/source_anchors'

# name, producer prefix, next consumer, expected recipient and input type.
CASES={
 'meeting':[
  ('private_request','PRIVATE_BOOKING','ACK_PRIVATE','Room','private_booking'),
  ('booking_authority','PRIVATE_BOOKING ACK_PRIVATE','SEND_CASE_LINK','HR','booking_request'),
  ('sanitization_input','SEND_ROSTER REQUEST_SANITIZED','COMPUTE_CONSTRAINTS','Sanitizer','request_constraints'),
  ('sanitization_result','SEND_ROSTER REQUEST_SANITIZED COMPUTE_CONSTRAINTS','SEND_CONSTRAINTS','HR','constraints'),
  ('room_constraints','PRIVATE_BOOKING ACK_PRIVATE SEND_CASE_LINK','RESERVE','Room','constraints'),
  ('planner_result','PRIVATE_BOOKING ACK_PRIVATE SEND_CASE_LINK RESERVE','DELIVER_RESULT','Planner','reservation')],
 'query':[
  ('vault_request','OPEN_VAULT','FETCH_COUNTS','Vault','join_request'),
  ('projection_request','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION','PROJECT_CITIES','Cassie','projection_request'),
  ('projection_result','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES','SEND_CITIES','Cassie','projection_ready'),
  ('counts','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES SEND_CITIES','JOIN','R','counts'),
  ('cities','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES SEND_CITIES','JOIN','R','cities'),
  ('vault_join_result','OPEN_VAULT FETCH_COUNTS FETCH_CITY_BUNDLE JOIN','SEND_JOIN_RESULT','Vault','joined_result'),
  ('broker_result','OPEN_VAULT FETCH_COUNTS FETCH_CITY_BUNDLE JOIN SEND_JOIN_RESULT','DELIVER_RESULT','R','joined_result')]
}


def prepare(kind,prefix):
    env=DynamicTask(kind,SOURCE,related=True,horizon=20);state=env.decision_start()
    for action in prefix.split():
        state,audit=env.transition(state,action)
        assert audit['verdict']=='VALID',(action,audit)
    return env,state


def finish(env,state):
    value=env.exact(state)
    for action in value['witness'] or []:
        state,audit=env.transition(state,action);assert audit['verdict']=='VALID'
    return state


def dependency_cases(kind):
    results=[]
    for name,prefix,action,recipient,typ in CASES[kind]:
      for intervention in ('identity','delete','redirect','tamper','blocked_delivery'):
        env,state=prepare(kind,prefix)
        target=next(m for m in state.ledger if m['recipient']==recipient and m['content']['type']==typ)
        if intervention in ('delete','blocked_delivery'):
            # Block at the producer/consumer boundary; preserve stage and caches
            # deliberately so they cannot silently bypass required delivery.
            state.ledger.remove(target)
        elif intervention=='redirect':target['recipient']='UnrelatedSink'
        elif intervention=='tamper':target['content']['unexpected']='modified payload'
        before=asdict(state);after,audit=env.transition(state,action)
        if intervention=='identity':
            assert audit['verdict']=='VALID';final=finish(env,after)
            assert final.success and not final.violation
        else:
            assert audit['verdict']=='UNKNOWN' and audit.get('task_evidence_rejected'),(name,intervention,audit)
            assert asdict(after)==before
        results.append({'task':kind,'input':name,'intervention':intervention,'consumer':action,
                        'before':before,'audit':audit,'after':asdict(after),'passed':True})
    return results


def computational_controls(kind):
    results=[]
    if kind=='meeting':
        prefix='PRIVATE_BOOKING ACK_PRIVATE SEND_CASE_LINK';consumer='RESERVE';typ='constraints';recipient='Room'
    else:
        prefix='OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES SEND_CITIES'
        consumer='JOIN';typ='counts';recipient='R'
    for mode in ('identity','trusted_changed_payload','poison_global_cache'):
        env,state=prepare(kind,prefix);target=next(m for m in state.ledger if m['recipient']==recipient and m['content']['type']==typ)
        if mode=='trusted_changed_payload':
            # Authorized producer counterfactual: reseal changed data so a
            # checksum rejection cannot masquerade as computational dependence.
            if kind=='meeting':
                c=next(c for c in target['content']['rows'] if c['type']=='cannot_use_room');c['room_id']='elm'
            else:target['content']['rows'][0]['gas_stations_count']+=100
            target['sha256']=env.evidence_hash(target)
        if mode=='poison_global_cache':state.result={'SHARED_CACHE':'must never be consumed'}
        after,audit=env.transition(state,consumer);assert audit['verdict']=='VALID',audit
        result=deepcopy(after.result)
        # Complete the known delivery path, even for a result rejected by oracle.
        after,audit=env.transition(after,'DELIVER_RESULT');assert audit['verdict']=='VALID'
        if mode=='trusted_changed_payload':assert result!=env.expected and not after.success
        else:assert result==env.expected and after.success
        results.append({'task':kind,'mode':mode,'delivered_input':target,'computed_result':result,
                        'task_success':after.success,'state':asdict(after),'passed':True})
    # Changing the diagnostic cache AFTER computation must not change delivery.
    env,state=prepare(kind,prefix);state,_=env.transition(state,consumer)
    state.result={'POISON':'no receipt fallback'}
    after,audit=env.transition(state,'DELIVER_RESULT')
    assert audit['verdict']=='VALID' and after.success and after.result==env.expected
    results.append({'task':kind,'mode':'poison_result_before_notify','state':asdict(after),'passed':True})
    return results


def acl_controls(kind):
    cases=(('HR','business_constraints','PRIVATE_BOOKING ACK_PRIVATE','SEND_CASE_LINK'),
           ('Room','public_schedule','PRIVATE_BOOKING ACK_PRIVATE SEND_CASE_LINK','RESERVE')) if kind=='meeting' else (
           ('Lina','company_gas_stations','OPEN_INDEX','FETCH_COUNTS'),
           ('JoinTool','company_id_map','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES SEND_CITIES','JOIN'))
    results=[]
    for actor,resource,prefix,action in cases:
        env,state=prepare(kind,prefix);env.source_acl[resource].remove(actor)
        after,audit=env.transition(state,action)
        assert audit['verdict']=='UNKNOWN' and after==state
        results.append({'task':kind,'actor':actor,'revoked_source':resource,'audit':audit,'passed':True})
    return results


def run(kind,output):
    results={'dependency':dependency_cases(kind),'computation':computational_controls(kind),'acl':acl_controls(kind)}
    results['all_passed']=all(r['passed'] for rows in results.values() for r in rows)
    results['scope']='canonical trusted simulator; delivery fault injection plus trusted producer value interventions; not a free-text or network security proof'
    Path(output).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'task':kind,'dependency_cases':len(results['dependency']),
                      'computation_controls':len(results['computation']),'acl_controls':len(results['acl']),'all_passed':results['all_passed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('output');a=p.parse_args();run(a.kind,a.output)
