"""Native environment feasibility and action dependencies, using annotated plans.

This is oracle action-list replay, not an agent or benchmark score reproduction.
"""
import sys,os,json,hashlib,time,threading,importlib.metadata
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];SOURCE=R/'tau2_source'
os.environ['PYTHON_DOTENV_DISABLED']='1'
sys.path[:0]=[str(ROOT/'.cache/tau2_deps_20261117'),str(SOURCE/'src')]
from tau2.domains.telecom.environment import get_environment,get_tasks

def execute(task,indices):
    env=get_environment();initial=task.initial_state
    if initial:env.set_state(initial.initialization_data,initial.initialization_actions,initial.message_history or [])
    trace=[];start=time.perf_counter()
    for index in indices:
        action=task.evaluation_criteria.actions[index];before=[env.get_db_hash(),env.get_user_db_hash()];error=None
        try:
            result=env.make_tool_call(action.name,requestor=action.requestor,**action.arguments);env.sync_tools()
        except Exception as exc:result=None;error=type(exc).__name__+': '+str(exc)
        after=[env.get_db_hash(),env.get_user_db_hash()]
        trace.append({'original_action_index':index,'role':action.requestor,'name':action.name,'arguments':action.arguments,'result':result,'error':error,'before_db_hashes':before,'after_db_hashes':after,'database_changed':before!=after})
    checks=[]
    for assertion in task.evaluation_criteria.env_assertions or []:
        try:passed=env.run_env_assertion(assertion,raise_assertion_error=False);error=None
        except Exception as exc:passed=False;error=type(exc).__name__+': '+str(exc)
        checks.append({'assertion':assertion.model_dump(mode='json'),'passed':passed,'error':error})
    return {'passed':bool(checks) and all(c['passed'] for c in checks) and not any(t['error'] for t in trace),'trace':trace,'assertions':checks,'seconds_after_initialization':time.perf_counter()-start,'mutating_tool_calls':sum(t['database_changed'] for t in trace),'policy_decisions':0,'communication_messages':0}

def run():
    out=R/'tau2_replay';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();manifest=json.loads((SOURCE/'SOURCE_MANIFEST.json').read_text())
    for item in manifest['files']:
        raw=(SOURCE/item['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['git_blob_sha']
    tasks=get_tasks();plan={'source_revision':manifest['revision'],'source_manifest_sha256':hashlib.sha256((SOURCE/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tasks':len(tasks),'task_ids':[t.id for t in tasks],'variants':['annotated_order','omit_each_action','reverse_order'],'scope':'oracle action-list replay; not a learned agent, conversation simulation or official benchmark score','python':sys.version,'dependencies':{n:importlib.metadata.version(n) for n in ('pydantic','loguru','docstring-parser','deepdiff','toml','addict','python-dotenv')},'max_trial_seconds':9.5}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2));summaries=[];max_seconds=0;trials=0
    for task in tasks:
        actions=task.evaluation_criteria.actions or [];indices=list(range(len(actions)));variants=[('annotated_order',indices)]+[(f'omit_{i}',indices[:i]+indices[i+1:]) for i in indices]+[('reverse_order',indices[::-1])];records=[]
        for name,order in variants:
            def timeout():
                (out/'TIMEOUT.json').write_text(json.dumps({'task':task.id,'variant':name,'seconds':9.5}));os._exit(124)
            timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter();record=execute(task,order);seconds=time.perf_counter()-start;timer.cancel();max_seconds=max(max_seconds,seconds);trials+=1;record.update(task_id=task.id,variant=name,total_seconds=seconds);records.append(record)
            with (out/'raw.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
        baseline=records[0];omissions=records[1:-1];summary={'task_id':task.id,'annotated_actions':len(actions),'baseline_pass':baseline['passed'],'mutating_calls':baseline['mutating_tool_calls'],'roles':sorted({a.requestor for a in actions}),'necessary_by_omission':sum(baseline['passed'] and not r['passed'] for r in omissions),'reverse_order_pass':records[-1]['passed']};summaries.append(summary)
        print(json.dumps(summary),flush=True)
    result={'tasks':len(tasks),'trials':trials,'baseline_pass':sum(r['baseline_pass'] for r in summaries),'max_annotated_actions':max(r['annotated_actions'] for r in summaries),'max_mutating_calls':max(r['mutating_calls'] for r in summaries),'two_role_plans':sum(len(r['roles'])==2 for r in summaries),'max_trial_seconds':max_seconds,'summaries':summaries,'raw_sha256':hashlib.sha256((out/'raw.jsonl').read_bytes()).hexdigest(),'proved_32_step_agent_collaboration':False,'learned_agent_evaluated':False,'natural_business_policy_validated':False}
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='summaries'}))

if __name__=='__main__':run()
