"""Fresh replay of frozen raw preferences and the observable-label gate.

This checks consistency with the final declared finite core; it is not an
independent semantic oracle or a proof beyond the enumerated case inventory.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from kci.contracts import Action,Trace,TraceStep,Verdict
from kci.continuation import ContinuationEvaluator,ReferenceSearch,SearchBudget,certify_preference
from kci.datasets import export_verified_preferences
from kci.multi_env import build_case,canonical
from kci.multi_candidates import MultiRuleCandidateGenerator
from kci.repair import DownstreamPolicyRepair
from kci.rewards import outcome_from_trace

DEPENDENCIES=('contracts','continuation','datasets','multi_env','multi_candidates','repair','rewards')
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]
def signature(row):return sha256('\n'.join(row[k] for k in ('prompt','chosen','rejected')).encode()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf-8')
def same(left,right):return canonical(left)==canonical(right)


def worker(root,filename):
    started=time.perf_counter();plan=read(root/'plan.json')
    for name,expected in plan['core_dependencies_sha256'].items():assert digest(name)==expected
    path=Path(plan['raw_dir'])/filename;assert digest(path)==plan['input_raw_sha256'][filename]
    data=read(path);spec=data['case_spec']
    env=build_case(spec['family'],spec['answer'],spec['variant'],spec['task_op'],
                   allow_protected=spec['allow_protected'],max_turns=spec['max_turns'])
    assert env.version==data['environment_version'] and env.expected_answer==data['expected_answer']
    original_hashes={k.replace('\\','/'):v for k,v in data['source_sha256'].items()}
    for name,expected in plan['core_dependencies_sha256'].items():assert original_hashes[name]==expected
    generator=MultiRuleCandidateGenerator();cce=ContinuationEvaluator(env,ReferenceSearch(env,generator,SearchBudget(3000,12)))
    state_cache={};report_cache={};prefix_steps=0
    def replay(prefix,lawful=True):
        nonlocal prefix_steps
        key=canonical(prefix)
        if lawful and key in state_cache:return state_cache[key]
        state=env.initial();steps=[]
        for value in prefix:
            action=Action.from_mapping(value);after,audit=env.transition(state,action);prefix_steps+=1
            assert audit.verdict==Verdict.VALID if lawful else audit.verdict in (Verdict.VALID,Verdict.VIOLATION)
            steps.append(TraceStep(state,action,audit,after));state=after
        if lawful:
            assert not state.any_violation and not state.done
            state_cache[key]=state;return state
        return Trace(tuple(steps),start_state=state if not steps else None)
    def evaluate(state,action):
        key=(state.signature(),action.key())
        if key not in report_cache:report_cache[key]=cce.evaluate(state,action)
        return report_cache[key]
    def report_row(report):
        ref=report.reference
        return {'action':asdict(report.action),'immediate':report.immediate.verdict.value,
            'evidence':asdict(report.immediate),'reference_status':ref.status.value if ref else None,
            'minimum_safe_total_cost':ref.best_success_cost if ref else None,
            'witness_actions':[asdict(s.action) for s in ref.best_trace.steps] if ref and ref.best_trace else None}
    state_proofs=[]
    for recorded in data['states']:
        state=replay(recorded['prefix']);prompt=env.observation(state,state.actor).as_prompt()
        assert prompt.encode()==recorded['prompt'].encode() and cce.state_id(state)==recorded['state_id']
        batch=generator.generate(env.observation(state,state.actor),12);assert batch.exhaustive
        reports=[evaluate(state,a) for a in batch.actions]
        assert same([report_row(r) for r in reports],recorded['actions'])
        directions=[]
        for a,b in combinations(reports,2):
            pref=certify_preference(a,b)
            if pref:directions.append([pref.chosen.as_json(),pref.rejected.as_json()])
        state_proofs.append({'case_id':spec['case_id'],'state_id':cce.state_id(state),'prompt':prompt,
                             'fresh_strict_directions':directions})
    pref_proofs=[]
    for row in data['preferences']:
        metadata=row['metadata'];state=replay(metadata['replay_prefix'])
        assert metadata['case_spec']==spec and same(metadata['world_configuration'],env.configuration())
        assert env.observation(state,state.actor).as_prompt().encode()==row['prompt'].encode()
        assert metadata['state_id']==cce.state_id(state)
        assert metadata['norm_version']==state.norm_version and metadata['task_oracle_version']==state.oracle_version
        chosen=Action.from_mapping(json.loads(row['chosen']));rejected=Action.from_mapping(json.loads(row['rejected']))
        assert chosen.as_json()==row['chosen'] and rejected.as_json()==row['rejected']
        good,bad=evaluate(state,chosen),evaluate(state,rejected)
        fresh=certify_preference(good,bad);assert fresh is not None
        assert fresh.chosen==chosen and fresh.rejected==rejected and fresh.source==metadata['source']
        for key,value in fresh.evidence.items():assert same(value,metadata[key]),key
        exported=export_verified_preferences([fresh])[0]
        assert all(exported[k]==row[k] for k in ('prompt','chosen','rejected'))
        original_by_action={canonical(v['action']):v for v in metadata['certification_reports']}
        for report in (good,bad):assert same(report_row(report),original_by_action[canonical(asdict(report.action))])
        pref_proofs.append({'raw_record_id':metadata['raw_record_id'],'case_id':spec['case_id'],
            'signature':signature(row),'state_id':cce.state_id(state),'source':fresh.source,
            'prompt_sha256':sha256(row['prompt'].encode()).hexdigest(),'direction_and_evidence_recertified':True})
    repairs=[];repairer=DownstreamPolicyRepair(cce,generator)
    for previous in data['repair_diagnostics']:
        trace=replay(previous['failed_actions'],lawful=False);assert outcome_from_trace(trace).safe_success is False
        result=repairer.repair(trace)
        assert (result.status,result.reason,result.failed_step)==(previous['status'],previous['reason'],previous['failed_step'])
        row=result.preference.to_dpo_record() if result.preference else None
        assert same(row,previous['preference'])
        repairs.append(signature(row) if row else None)
    save(root/'cases'/filename,{'case_id':spec['case_id'],'raw_input_sha256':digest(path),
        'fresh_state_proofs':state_proofs,'fresh_preference_proofs':pref_proofs,'fresh_repair_signatures':repairs,
        'replayed_prefix_steps':prefix_steps,'unique_evaluated_actions':len(report_cache),
        'seconds':time.perf_counter()-started,'core_dependencies_sha256':plan['core_dependencies_sha256']})
    print(json.dumps({'case':spec['case_id'],'preferences':len(pref_proofs),'states':len(state_proofs),'repairs':len(repairs),'seconds':time.perf_counter()-started}))


def assemble(root):
    plan=read(root/'plan.json');dataset=Path(plan['dataset_dir']);audit=Path(plan['audit_dir'])
    for path,expected in plan['release_inputs_sha256'].items():assert digest(path)==expected
    groups=defaultdict(list);verified={};repair_signatures=set();all_raw=[]
    for filename in plan['input_raw_sha256']:
        result=read(root/'cases'/filename)
        for state in result['fresh_state_proofs']:groups[state['prompt']].append(state)
        for pref in result['fresh_preference_proofs']:
            assert pref['raw_record_id'] not in verified;verified[pref['raw_record_id']]=pref
        repair_signatures.update(s for s in result['fresh_repair_signatures'] if s)
        all_raw.extend(read(Path(plan['raw_dir'])/filename)['preferences'])
    raw_by_signature=defaultdict(list);gate_results={}
    for row in all_raw:raw_by_signature[signature(row)].append(row)
    for key,members in raw_by_signature.items():
        row=members[0];orientation=[row['chosen'],row['rejected']];states=groups[row['prompt']]
        support=[orientation in s['fresh_strict_directions'] for s in states]
        gate_results[key]={'signature':key,'compatible_states':len(states),'uniformly_supported':bool(states) and all(support),
            'unsupported_states':[{'case_id':s['case_id'],'state_id':s['state_id']} for s,ok in zip(states,support) if not ok]}
    excluded={k for k,v in gate_results.items() if not v['uniformly_supported']}
    expected_excluded={r['preference_signature'] for r in rows(audit/'inconsistent_preferences.jsonl')}
    assert excluded==expected_excluded
    accepted=rows(dataset/'preferences.jsonl');quarantine=rows(dataset/'quarantine.jsonl')
    assert {signature(r) for r in quarantine}==excluded
    assert {signature(r) for r in accepted}==set(raw_by_signature)-excluded
    assert len(accepted)==len({signature(r) for r in accepted})
    accepted_checks=[]
    for row in accepted:
        key=signature(row);m=row['metadata'];sources=raw_by_signature[key]
        assert m['preference_id']==key and m['observation_class_id']==sha256(row['prompt'].encode()).hexdigest()
        assert m['observation_audit_sha256']==digest(audit/'summary.json')
        assert m['observable_equivalence_gate']=='uniform_strict_preference_in_enumerated_compatible_states'
        assert sorted(m['contributing_raw_ids'])==sorted(s['metadata']['raw_record_id'] for s in sources)
        assert sorted(m['contributing_case_ids'])==sorted(s['metadata']['case_spec']['case_id'] for s in sources)
        assert all(verified[identifier]['signature']==key for identifier in m['contributing_raw_ids'])
        assert ('repair' in m['training_sources'])==(key in repair_signatures)
        accepted_checks.append({'preference_id':key,'raw_source_count':len(sources),'compatible_states':gate_results[key]['compatible_states'],
            'fresh_source_and_observation_gate_valid':True})
    quarantined_checks=[]
    for row in quarantine:
        identifier=row['metadata']['raw_record_id'];assert verified[identifier]['direction_and_evidence_recertified']
        quarantined_checks.append({'raw_record_id':identifier,**gate_results[signature(row)],
            'status':'state-valid_but_observation-nonidentifiable; remain excluded from hard supervision'})
    save(root/'accepted_release_checks.json',accepted_checks);save(root/'quarantine_replay_checks.json',quarantined_checks)
    summary={'case_workers':len(plan['input_raw_sha256']),'fresh_raw_preferences':len(verified),
        'fresh_saved_nonterminal_states':sum(len(g) for g in groups.values()),'exact_prompt_classes':len(groups),
        'raw_preference_unique_signatures':len(raw_by_signature),'uniform_observation_gate_rejected_signatures':len(excluded),
        'state_valid_quarantined_raw_rows':len(quarantined_checks),'accepted_unique_preferences':len(accepted_checks),
        'fresh_repair_diagnostics':sum(len(read(root/'cases'/f)['fresh_repair_signatures']) for f in plan['input_raw_sha256']),
        'release_raw_sources_and_gate_all_passed':True,'gate_matches_independent_original_audit':True,
        'core_dependencies_sha256':plan['core_dependencies_sha256'],'scope':'fresh final-core consistency replay; finite declared inventory, not independent semantic oracle',
        'unrelated_adapter_changes_used_as_evidence_dependency':False,'inputs_unchanged':True}
    for path,expected in plan['release_inputs_sha256'].items():assert digest(path)==expected
    for filename,expected in plan['input_raw_sha256'].items():assert digest(Path(plan['raw_dir'])/filename)==expected
    for path,expected in plan['core_dependencies_sha256'].items():assert digest(path)==expected
    save(root/'summary.json',summary);print(json.dumps(summary))


def run(raw,dataset,audit,root):
    root.mkdir(parents=True,exist_ok=False);files=sorted(raw.glob('*.json'));assert len(files)==112
    deps={f'kci/{name}.py':digest(f'kci/{name}.py') for name in DEPENDENCIES}
    release_paths=[dataset/n for n in ('preferences.jsonl','quarantine.jsonl','summary.json')]+[audit/n for n in ('summary.json','inconsistent_preferences.jsonl')]
    plan={'raw_dir':str(raw.resolve()),'dataset_dir':str(dataset.resolve()),'audit_dir':str(audit.resolve()),
        'input_raw_sha256':{p.name:digest(p) for p in files},'core_dependencies_sha256':deps,
        'release_inputs_sha256':{str(p.resolve()):digest(p) for p in release_paths},'hard_timeout_seconds':9.5,
        'replay_script_sha256':digest(__file__),'note':'independent fresh process invocation; reuses frozen final core'}
    save(root/'plan.json',plan);source=root/'source';source.mkdir()
    for path in list(deps)+[__file__]:shutil.copy2(path,source/Path(path).name)
    receipts=[]
    for name in [p.name for p in files]+['ASSEMBLE']:
        started=time.perf_counter();command=[sys.executable,'-B','-m','experiments.final_dataset_replay','--root',str(root.resolve()),'--worker',name]
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=9.5)
            receipt={'worker':name,'returncode':process.returncode,'seconds':time.perf_counter()-started,
                     'hard_timeout_seconds':9.5,'stdout':process.stdout,'stderr':process.stderr}
        except subprocess.TimeoutExpired as error:
            receipt={'worker':name,'timeout':True,'seconds':time.perf_counter()-started,'hard_timeout_seconds':9.5,
                     'stdout':str(error.stdout),'stderr':str(error.stderr)}
        receipts.append(receipt);save(root/'worker_receipts.json',receipts)
        if receipt.get('returncode')!=0:raise RuntimeError('fresh replay failed; preserve evidence')
    summary=read(root/'summary.json');summary.update({'successful_workers':len(receipts),'hard_timeout_seconds':9.5,
        'max_worker_seconds':max(r['seconds'] for r in receipts)})
    save(root/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw');p.add_argument('--dataset');p.add_argument('--audit')
    p.add_argument('--root',required=True);p.add_argument('--worker');a=p.parse_args();root=Path(a.root)
    if a.worker=='ASSEMBLE':assemble(root)
    elif a.worker:worker(root,a.worker)
    else:run(Path(a.raw),Path(a.dataset),Path(a.audit),root)
