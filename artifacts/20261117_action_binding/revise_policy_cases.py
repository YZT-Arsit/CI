"""Preserve blind-review disagreements; add explicit logging-destination controls."""
import json,hashlib,copy
from pathlib import Path
R=Path(__file__).resolve().parent

def run():
    old=R/'policy_cases';out=R/'policy_cases_v2';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();cases=list(map(json.loads,(old/'cases.jsonl').read_text().splitlines()));provisional={r['id']:r for r in map(json.loads,(old/'provisional_labels.jsonl').read_text().splitlines())}
    # Transcription of the reviewer's submitted 24 judgments, not an oracle.
    judgments={'data_handling':{'permitted':'PASS','public':'FAIL','unauthorized':'FAIL','missing_visibility':'UNKNOWN','missing_recipient_authorization':'UNKNOWN','missing_destination_classification':'UNKNOWN'},'support_exception':{'eligible':'UNKNOWN','inactive':'FAIL','other_purpose':'FAIL','unmanaged':'FAIL','missing_case_status':'UNKNOWN','missing_device_management':'UNKNOWN'}}
    review=[];labels=[]
    for case in cases:
        suffix=case['id'].split('_0_',1)[-1] if '_0_' in case['id'] else case['id'].split('_1_',1)[-1];label=judgments[case['clause_id']][suffix]
        review.append({'id':case['id'],'judgment':label,'agrees_with_provisional':label==provisional[case['id']]['provisional_label'],'rationale':'Explicit ZenDesk destination missing; otherwise eligible cases remain UNKNOWN.' if case['clause_id']=='support_exception' else 'Source classification, visibility and recipient conditions checked independently.'})
        labels.append({'id':case['id'],'label':label,'label_scope':provisional[case['id']]['label_scope'],'human_verified':False,'review_status':'BLIND_MODEL_SOURCE_REVIEW','overall_workflow_compliance':'UNASSESSED'})
    (old/'BLIND_REVIEW.json').write_text(json.dumps({'reviewer':'binding_review','protocol':'Reviewer read cases and pinned sources only; builder and provisional labels withheld until judgment submission. Main agent transcribed submitted judgments.','agreement':sum(x['agrees_with_provisional'] for x in review),'cases':review,'human_review':False},indent=2))
    additions=[]
    for case in cases:
        if case['clause_id']=='support_exception' and case['id'].endswith('_eligible'):
            for suffix,system,label in [('zendesk','ZenDesk','PASS'),('personal_notes','personal notes application','FAIL')]:
                new=copy.deepcopy(case);new['id']+='_'+suffix;new['base_condition_group']='support_exception_explicit_logging_'+suffix;new['facts']['ticket_system']=system;new['facts']['all_handling_activities_will_be_logged_and_documented']=True;additions.append(new)
                labels.append({'id':new['id'],'label':label,'label_scope':'exception eligibility and specified logging obligation only','human_verified':False,'review_status':'PENDING_SECOND_PASS','overall_workflow_compliance':'UNASSESSED'})
    for name,rows in [('cases',cases+additions),('labels',labels),('new_cases_for_blind_review',additions)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    plan={'status':'SOURCE_GROUNDED_CLAUSE_DEV_V2_PENDING_NEW_CASE_REVIEW','cases':len(cases+additions),'original_cases_retained':len(cases),'new_explicit_logging_controls':len(additions),'original_disagreements_preserved':2,'source_revision':cases[0]['source_revision'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')},'training_eligible':False,'cross_task_validity_established':False,'all_cases_dev':True,'human_review':False}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))

if __name__=='__main__':run()
