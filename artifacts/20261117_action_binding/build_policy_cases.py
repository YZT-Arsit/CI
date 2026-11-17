"""Authored clause-comprehension DEV cases, separated from provisional labels."""
import json,hashlib,copy
from pathlib import Path
R=Path(__file__).resolve().parent

def run():
    out=R/'policy_cases';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();pin=json.loads((R/'SOURCE_PIN.json').read_text());cases=[];labels=[]
    definitions=[
      ('data_handling',0,['internal_bug_discussion','support_log_collaboration'],
       {'organization':'GitLab','actor':'support team member','data_classification':'ORANGE','destination_classification':'RED','visibility':'confidential','recipient_authorized':True,'classifications_verified_for_this_hypothetical_case':True,'other_relevant_requirements_verified':True},
       [('permitted',{},'PASS'),('public',{'visibility':'public'},'FAIL'),('unauthorized',{'recipient_authorized':False},'FAIL'),('missing_visibility',{'visibility':None},'UNKNOWN'),('missing_recipient_authorization',{'recipient_authorized':None},'UNKNOWN'),('missing_destination_classification',{'destination_classification':None,'classifications_verified_for_this_hypothetical_case':False},'UNKNOWN')],
       'Does this proposed posting satisfy the cited data-classification, visibility and recipient-authorization requirements? Judge only these requirements; do not infer missing facts.'),
      ('support_exception',1,['local_log_reproduction','local_database_reproduction'],
       {'organization':'GitLab','actor':'team member','data_classification':'RED customer data','active_customer_support_case':True,'purpose':'customer support','gitlab_managed_laptop':True,'separate_exception_approval':False,'question_time':'before migration','handling_will_be_logged_in_ticket':True,'data_will_be_deleted_immediately_after_case_resolution':True},
       [('eligible',{},'PASS'),('inactive',{'active_customer_support_case':False},'FAIL'),('other_purpose',{'purpose':'internal research unrelated to customer support'},'FAIL'),('unmanaged',{'gitlab_managed_laptop':False},'FAIL'),('missing_case_status',{'active_customer_support_case':None},'UNKNOWN'),('missing_device_management',{'gitlab_managed_laptop':None},'UNKNOWN')],
       'Is this proposed local migration eligible under the cited customer-support exception? Judge exception eligibility only, not all organizational policy. Future logging and deletion are promised obligations, not completed actions.')]
    for clause,source_index,tasks,base,variants,question in definitions:
        source=pin['files'][source_index]
        for template,task in enumerate(tasks):
            for variant,changes,label in variants:
                facts=copy.deepcopy(base);facts.update(changes);identifier=f'{clause}_{template}_{variant}'
                cases.append({'id':identifier,'clause_id':clause,'base_condition_group':clause+'_'+variant,'task_template':task,'facts':facts,'question':question,'source_url':source['permalink'],'source_sha256':source['sha256'],'source_revision':pin['revision'],'evidence_type':'authored hypothetical case tied to real organization clause','answers_allowed':['PASS','FAIL','UNKNOWN']})
                labels.append({'id':identifier,'provisional_label':label,'label_scope':'named clause requirements or exception eligibility only','overall_workflow_compliance':'UNASSESSED','human_verified':False,'independent_review_status':'PENDING'})
    for name,rows in [('cases',cases),('provisional_labels',labels)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    plan={'status':'PROVISIONAL_CLAUSE_DEV_ONLY','cases':len(cases),'base_condition_groups':len({c['base_condition_group'] for c in cases}),'clauses':2,'task_templates':4,'source_revision':pin['revision'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'cases.jsonl',out/'provisional_labels.jsonl')},'training_eligible':False,'cross_task_validity_established':False,'real_business_execution':False,'all_cases_dev':True,'split_requirement':'Keep shared base_condition_group together in any future random split; template transfer is not source-level holdout.'}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))

if __name__=='__main__':run()
