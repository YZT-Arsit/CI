"""Native role-boundary checks and annotated replay; zero learned decisions."""
import hashlib,time,threading
from session import *
from tau2.domains.telecom.environment import get_tasks

def main():
    assert not (R/'PLAN.json').exists();manifest=json.loads((SOURCE/'SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        data=(SOURCE/row['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha']
    tasks=get_tasks();plan={'scope':'role-local transport and oracle replay validation, not a learned-agent or32-step result','tasks':len(tasks),'source_revision':manifest['revision'],'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'session.py',Path(__file__)]},'max_trial_seconds':9.5,'policy_source':'native benchmark telecom rules; no GitLab rules transplanted'};(R/'PLAN.json').write_text(json.dumps(plan,indent=2));records=[];assistant_reference=None
    for task in tasks:
        def timeout():
            (R/'TIMEOUT.json').write_text(json.dumps({'task_id':task.id}));os._exit(124)
        timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter();session=RoleSession(task);assistant=session.view('assistant');user=session.view('user')
        if assistant_reference is None:assistant_reference=assistant
        assert assistant==assistant_reference  # No diagnostic task ID, ticket, hidden state or evaluator leakage.
        altered=task.model_copy(deep=True);altered.id='HIDDEN_TASK_CANARY';altered.description.notes='HIDDEN_EVALUATOR_CANARY';altered.evaluation_criteria.actions=[];alternate=RoleSession(altered);assert alternate.view('assistant')==assistant and alternate.view('user')==user
        hidden=task.model_copy(deep=True);hidden.user_scenario.instructions='PRIVATE_USER_SCENARIO_CANARY';private=RoleSession(hidden);assert private.view('assistant')==assistant and private.view('user')!=user
        receiver_before=session.view('assistant');session.message('user','LOCAL_TEST_MESSAGE',deliver=False);assert session.view('assistant')==receiver_before;session.message('user','DELIVERED_TEST_MESSAGE');assert session.view('assistant')['history'][-1]['content']=='DELIVERED_TEST_MESSAGE'
        original=session.view('assistant');edited=session.view('assistant');edited['history'].append({'content':'COPY_MUTATION'});assert session.view('assistant')==original
        tool_private=0;wrong_role_checks=0;trace=[]
        for action in task.evaluation_criteria.actions or []:
            role=action.requestor;other='user' if role=='assistant' else 'assistant';before=session.view(other);entry=session.tool(role,action.name,action.arguments);assert entry['error'] is None;assert session.view(other)==before;tool_private+=1;trace.append(entry)
            other_names={s['function']['name'] for s in session.schemas[other]}
            if action.name not in other_names:
                hashes=[session.env.get_db_hash(),session.env.get_user_db_hash()];own_before=session.view(role);rejected=session.tool(other,action.name,action.arguments);assert rejected['error'] and hashes==[session.env.get_db_hash(),session.env.get_user_db_hash()] and session.view(role)==own_before;wrong_role_checks+=1
        assertions=[session.env.run_env_assertion(a,raise_assertion_error=False) for a in task.evaluation_criteria.env_assertions or []];assert assertions and all(assertions);timer.cancel();record={'task_id':task.id,'passed_native_assertions':True,'private_tool_results_checked':tool_private,'wrong_role_calls_rejected':wrong_role_checks,'assistant_initial_view_identical_across_tasks':True,'hidden_metadata_and_user_scenario_noninterference':True,'message_drop_and_delivery_checked':True,'view_copy_isolated':True,'counters':session.counters(),'seconds':time.perf_counter()-start,'model_decisions':0,'trace':trace};records.append(record)
        with (R/'raw.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    result={'tasks':len(records),'native_replay_success':sum(r['passed_native_assertions'] for r in records),'private_tool_result_checks':sum(r['private_tool_results_checked'] for r in records),'wrong_role_rejections':sum(r['wrong_role_calls_rejected'] for r in records),'max_trial_seconds':max(r['seconds'] for r in records),'model_decisions':0,'message_tests_are_synthetic_canaries':True,'task_success_under_message_omission_tested':False,'learned_collaboration_established':False,'whole_goal_complete':False,'raw_sha256':hashlib.sha256((R/'raw.jsonl').read_bytes()).hexdigest()};(R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
