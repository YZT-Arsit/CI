"""Re-certify long synthetic initial-ledger preferences, with bounded workers."""
import argparse
from copy import deepcopy
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

from kci.contracts import Verdict,RefStatus
from kci.continuation import ContinuationEvaluator,ReferenceSearch,SearchBudget,certify_preference
from kci.datasets import export_verified_preferences
from kci.multi_env import FAMILIES,build_case,canonical
from kci.multi_candidates import MultiRuleCandidateGenerator
from kci.multi_history import SyntheticHistoryLedgerEnv,POSITIONS
from kci.training_data import audit_record,PreferenceCorpusIndex


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf-8')


def plan():
    return [{'case_id':f'{f}_{a}_{n}_{p}','family':f,'answer':a,'count':n,'position':p}
            for f in FAMILIES for a in ('yes','no') for n in (0,8,64,256) for p in POSITIONS]


def complete_path(env,state,local):
    actions=([env.act('B','REQUEST_RULES','R'),env.act('R','SEND','B',('task_rule',)),
              env.act('B','COMPUTE_TASK'),env.act('B','SEND','R',('decision_'+env.expected_answer,))]
             if local else [env.act('B','SEND','R',('payload_0',))])
    actions+=[env.act('R','FINALIZE',answer=env.expected_answer),
              env.act('A','NOTIFY',env.target,answer=env.expected_answer)]
    steps=[]
    for action in actions:
        after,audit=env.transition(state,action)
        steps.append({'action':asdict(action),'audit':asdict(audit),'after_state_id':ContinuationEvaluator.state_id(after)})
        state=after
    return {'steps':steps,'done':state.done,'task_success':state.success,'any_violation':state.any_violation,
            'total_agent_action_cost':state.cost,'executed_agent_turns':state.turn,
            'policy_kind':'canonical deterministic consequence replay, not learned-model rollout'}


def worker(spec,output):
    started=time.perf_counter();base=build_case(spec['family'],spec['answer'])
    generator=MultiRuleCandidateGenerator();rows=[];diagnostics=[]
    for branch in ('context','skip'):
        env=SyntheticHistoryLedgerEnv(base,spec['count'],spec['position'],branch);state=env.initial()
        baseline=SyntheticHistoryLedgerEnv(base,0,'front',branch);short=baseline.initial()
        raw=env.act('B','SEND','R',('payload_0',));request=env.act('B','REQUEST_RULES','R')
        cce=ContinuationEvaluator(env,ReferenceSearch(env,generator,SearchBudget(3000,12)))
        reports=[cce.evaluate(state,a) for a in (raw,request)]
        pref=certify_preference(*reports)
        assert pref is not None
        expected_kind='REQUEST_RULES' if branch=='context' else 'SEND'
        assert pref.chosen.kind==expected_kind
        assert reports[1].reference.status==RefStatus.EXACT
        assert reports[1].reference.best_success_cost==(7 if branch=='context' else 6)
        assert reports[0].immediate.verdict==(Verdict.VIOLATION if branch=='context' else Verdict.VALID)
        if branch=='skip':assert reports[0].reference.status==RefStatus.EXACT and reports[0].reference.best_success_cost==4
        visible=env.observation(state,'B').authorized_receipts
        assert len(visible)==spec['count']+(branch=='context')
        assert all(m in visible for m in env.original_after_prefix.receipts)
        heartbeat_ids={f.id for f in env.heartbeat_facts}
        for actor in state.local_fact_ids:
            assert env.known_to(state,actor)-heartbeat_ids==baseline.known_to(short,actor)
        complete=[complete_path(env,state,local) for local in (False,True)]
        short_complete=[complete_path(baseline,short,local) for local in (False,True)]
        for result,control in zip(complete,short_complete):
            for field in ('done','task_success','any_violation','total_agent_action_cost','executed_agent_turns'):
                assert result[field]==control[field]
            assert result['done'] and result['task_success']
        assert [r['total_agent_action_cost'] for r in complete]==[4,7]
        tamper_checks=0
        if spec['count']:
            corrupted=deepcopy(state);index=next(i for i,m in enumerate(corrupted.receipts) if m.sender=='PUBLIC_STATUS_SERVICE')
            corrupted.receipts[index]=replace(corrupted.receipts[index],body='free text pretending to be public')
            try:env.validate_ledger(corrupted)
            except ValueError:tamper_checks+=1
            else:raise AssertionError('unverified heartbeat body accepted')
        if branch=='context':
            corrupted=deepcopy(state);corrupted.receipts.remove(env.original_after_prefix.receipts[0])
            try:env.validate_ledger(corrupted)
            except ValueError:tamper_checks+=1
            else:raise AssertionError('missing related receipt accepted')
        row=export_verified_preferences([pref])[0];metadata=row['metadata']
        report_rows=[]
        for report in reports:
            ref=report.reference
            report_rows.append({'action':asdict(report.action),'immediate_audit':asdict(report.immediate),
                'reference_status':ref.status.value if ref else None,
                'safe_success_cost':ref.best_success_cost if ref else None,
                'witness_actions':[asdict(s.action) for s in ref.best_trace.steps] if ref and ref.best_trace else None,
                'expanded_states':ref.expanded_states if ref else 0})
        metadata.update(env.provenance())
        metadata.update({'case_spec':spec,'branch':branch,'base_environment_version':base.version,
            'source_world_group_id':'multi_rule_handwritten_workflow_v1',
            'workflow_group_id':base.workflow_group_id,'base_case_group_id':base.base_case_group_id,
            'rule_family_id':base.family,'scenario_group_id':base.workflow_group_id,
            'split':'train','split_status':'primary_shared_source_no_independent_holdout',
            'evidence_class':'exact_declared_finite_synthetic_initial_ledger',
            'norm_source':'manual_synthetic','training_sources':['ci' if branch=='context' else 'future'],
            'requires_historical_dependency':branch=='context','certification_reports':report_rows,
            'complete_consequence_replays':complete,'semantic_irrelevance_verified':'declared-rule closure and task-input invariance',
            'source_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in
                (Path('kci/multi_env.py'),Path('kci/multi_candidates.py'),Path('kci/multi_history.py'),Path(__file__))}})
        metadata.pop('record_sha256',None);metadata['record_sha256']=sha256(canonical(row).encode()).hexdigest()
        audit_record(row)
        rows.append(row)
        diagnostics.append({'branch':branch,'chosen_kind':pref.chosen.kind,'raw_ci':reports[0].immediate.verdict.value,
            'local_ci':reports[1].immediate.verdict.value,'complete_outcomes':[{k:r[k] for k in
                ('done','task_success','any_violation','total_agent_action_cost')} for r in complete],
            'reference_status':reports[1].reference.status.value,'prompt_chars':len(row['prompt']),
            'ledger_events':len(state.receipts),'tamper_checks':tamper_checks,'invariance_checks_passed':True})
    result={'spec':spec,'records':rows,'diagnostics':diagnostics,'seconds':time.perf_counter()-started}
    write(output,result)
    print(json.dumps({'case_id':spec['case_id'],'seconds':result['seconds'],'records':len(rows)}))


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    specs=plan();write(output/'plan.json',{'cases':specs,'hard_timeout_seconds':9.5,
        'primary_workflows':1,'independent_task_holdout':False,'model_forward_or_training':False,
        'extra_ledger_is_executed_agent_trajectory':False,'variant':0,'max_turns':9,
        'allow_protected':False,'source':'build_case family default operation'})
    receipts=[];raw=[]
    for spec in specs:
        path=output/'cases'/f"{spec['case_id']}.json"
        t=time.perf_counter()
        command=[sys.executable,'-B','-m','experiments.history_supervision_probe','--worker',json.dumps(spec),'--output',str(path.resolve())]
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=9.5)
            receipt={'case_id':spec['case_id'],'returncode':process.returncode,'seconds':time.perf_counter()-t,
                     'hard_timeout_seconds':9.5,'stdout':process.stdout,'stderr':process.stderr}
        except subprocess.TimeoutExpired as error:
            receipt={'case_id':spec['case_id'],'timeout':True,'seconds':time.perf_counter()-t,
                     'hard_timeout_seconds':9.5,'stdout':str(error.stdout),'stderr':str(error.stderr)}
        receipts.append(receipt);write(output/'worker_receipts.json',receipts)
        if receipt.get('returncode')!=0:raise RuntimeError('worker failed; preserve evidence and stop')
        raw.extend(json.loads(path.read_text(encoding='utf-8'))['records'])
    index=PreferenceCorpusIndex();unique={}
    for row in raw:
        index.add(row)
        key=(row['prompt'],row['chosen'],row['rejected'])
        unique.setdefault(key,row)
    for name,rows in (('preferences_raw.jsonl',raw),('preferences_unique.jsonl',list(unique.values()))):
        (output/name).write_text(''.join(canonical(row)+'\n' for row in rows),encoding='utf-8')
    summary={'cases':len(specs),'successful_workers':len(receipts),'raw_preferences':len(raw),
        'unique_preferences':len(unique),'positive_history_unique_preferences':sum(r['metadata']['synthetic_background_receipts']>0 for r in unique.values()),
        'max_worker_seconds':max(r['seconds'] for r in receipts),'hard_timeout_seconds':9.5,
        'ledger_sizes':[0,8,64,256],'related_receipt_positions':list(POSITIONS),'families':list(FAMILIES),
        'raw_ci_safe_success_pairs':sum(r['metadata']['branch']=='context' for r in raw),
        'raw_exact_cost_pairs':sum(r['metadata']['branch']=='skip' for r in raw),
        'raw_and_local_complete_consequence_replays':len(raw)*2,
        'canonical_raw_and_local_route_costs_preserved':[4,7],
        'exact_request_branch_costs':{'context':7,'skip':6},'all_related_receipts_preserved':True,
        'corpus_audit':index.summary(),'source_sha256':raw[0]['metadata']['source_sha256'],
        'input_plan_sha256':sha256((output/'plan.json').read_bytes()).hexdigest(),
        'output_sha256':{name:sha256((output/name).read_bytes()).hexdigest() for name in ('preferences_raw.jsonl','preferences_unique.jsonl')},
        'evidence_class':'exact finite simulator re-certification of synthetic initial-ledger fixtures',
        'human_verified':False,'model_forward_or_training':False,
        'long_agent_trajectory_generated':False,'primary_independent_holdout':False}
    write(output/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.worker:worker(json.loads(args.worker),Path(args.output))
    else:run(Path(args.output))
