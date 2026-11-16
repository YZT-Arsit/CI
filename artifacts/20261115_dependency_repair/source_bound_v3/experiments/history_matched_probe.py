"""Equal-event, equal-character related/unrelated long-ledger controls.

The pinned real tokenizer measures token equality; no padding or truncation is
used to manufacture equality. These are synthetic initial-ledger fixtures.
"""
import argparse
from dataclasses import asdict,replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from kci.contracts import Verdict,RefStatus
from kci.continuation import ContinuationEvaluator,ReferenceSearch,SearchBudget,certify_preference
from kci.datasets import export_verified_preferences
from kci.multi_env import FAMILIES,build_case,canonical,canonical_source_text,MultiRuleCIEnv
from kci.multi_candidates import MultiRuleCandidateGenerator
from kci.multi_history import SyntheticHistoryLedgerEnv,POSITIONS
from kci.training_data import PinnedTokenizerAssets,TrainingSequenceAdapter,PreferenceCorpusIndex
from experiments.history_supervision_probe import complete_path

PROTOCOL='matched_initial_ledger_supervision_v2'
ASSET_MANIFEST=Path('../artifacts/20261112_cpu_pilot/tokenization/config.json')
ASSET_MANIFEST_SHA='e1242b65a53ea00e4747af4af876f7c867ed53510e80ff0f8edc4f3911012d21'
SOURCE_PATHS=('kci/contracts.py','kci/continuation.py','kci/datasets.py','kci/multi_env.py',
              'kci/multi_candidates.py','kci/multi_history.py','kci/training_data.py',
              'experiments/history_supervision_probe.py','experiments/history_matched_probe.py')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf-8')


def specifications():
    return [{'case_id':f'{f}_{a}_{n}_{p}','family':f,'answer':a,'count':n,'position':p}
            for f in FAMILIES for a in ('yes','no') for n in (0,8,64,256)
            for p in (('front',) if n==0 else POSITIONS)]


def unrelated_control(base):
    sources={k:f for k,f in base.facts.items() if k.startswith(('context_','payload_')) or k=='notification_target'}
    key='context_1' if base.family=='alias_chain_3' else 'context_0'
    original=sources[key]
    if base.family=='identity_attribute':changed=replace(original,subject='Record001')
    elif base.family=='alias_chain_3':changed=replace(original,subject='Employee001')
    else:
        value=json.loads(original.value)
        if base.family=='aggregate_difference':value['period']='period_2'
        else:value['scope']='complete_candidate_registry_v2'
        changed=replace(original,value=canonical(value))
    sources[key]=replace(changed,text=canonical_source_text(changed))
    env=MultiRuleCIEnv(base.family,sources,json.loads(base.facts['task_rule'].value),
                       world_id=base.world_id+':unrelated_context',max_turns=base.max_turns)
    assert env.facts['payload_0']==base.facts['payload_0']
    assert env.facts['task_rule']==base.facts['task_rule'] and env.expected_answer==base.expected_answer
    assert len(sources[key].text)==len(original.text)
    return env,{'source_id':key,'original':asdict(original),'changed':asdict(sources[key]),
                'intervention':'one declared join/scope key changed; payload and task criterion unchanged'}


def worker(spec,output):
    started=time.perf_counter();base=build_case(spec['family'],spec['answer'])
    other,intervention=unrelated_control(base)
    adapter=TrainingSequenceAdapter(PinnedTokenizerAssets.from_manifest(ASSET_MANIFEST,
        expected_manifest_sha256=ASSET_MANIFEST_SHA))
    source_hashes={name:sha256(Path(name).read_bytes()).hexdigest() for name in SOURCE_PATHS}
    group=PROTOCOL+':'+spec['case_id'];records=[];sides=[];encoded=[]
    for relation,source in (('related',base),('unrelated',other)):
        env=SyntheticHistoryLedgerEnv(source,spec['count'],spec['position'],'context');state=env.initial()
        cce=ContinuationEvaluator(env,ReferenceSearch(env,MultiRuleCandidateGenerator(),SearchBudget(3000,12)))
        raw=env.act('B','SEND','R',('payload_0',));request=env.act('B','REQUEST_RULES','R')
        reports=[cce.evaluate(state,a) for a in (raw,request)]
        preference=certify_preference(*reports);assert preference is not None
        expected=Verdict.VIOLATION if relation=='related' else Verdict.VALID
        assert reports[0].immediate.verdict==expected
        assert preference.chosen.kind==('REQUEST_RULES' if relation=='related' else 'SEND')
        assert reports[1].reference.status==RefStatus.EXACT
        assert reports[1].reference.best_success_cost==(7 if relation=='related' else 6)
        if relation=='unrelated':
            assert reports[0].reference.status==RefStatus.EXACT and reports[0].reference.best_success_cost==4
        replays=[complete_path(env,state,local) for local in (False,True)]
        assert all(r['done'] and r['task_success'] for r in replays)
        assert [r['total_agent_action_cost'] for r in replays]==[4,7]
        assert [r['any_violation'] for r in replays]==[relation=='related',False]
        row=export_verified_preferences([preference])[0];metadata=row['metadata']
        metadata.update(env.provenance())
        metadata.update({'matched_protocol':PROTOCOL,'matched_pair_group_id':group,'relation':relation,
            'case_spec':spec,'intervention':intervention,'base_environment_version':source.version,
            'source_world_group_id':'multi_rule_handwritten_workflow_v1','workflow_group_id':base.workflow_group_id,
            'base_case_group_id':base.base_case_group_id,'rule_family_id':base.family,
            'scenario_group_id':base.workflow_group_id,'split':'train',
            'split_status':'primary_shared_source_no_independent_holdout','norm_source':'manual_synthetic',
            'evidence_class':'exact_declared_finite_synthetic_initial_ledger_matched_context',
            'training_sources':['ci' if relation=='related' else 'future'],'requires_historical_dependency':True,
            'source_sha256':source_hashes,'complete_consequence_replays':replays,
            'certification_reports':[{'action':asdict(r.action),'immediate_audit':asdict(r.immediate),
                'reference_status':r.reference.status.value if r.reference else None,
                'safe_success_cost':r.reference.best_success_cost if r.reference else None,
                'witness_actions':[asdict(s.action) for s in r.reference.best_trace.steps]
                if r.reference and r.reference.best_trace else None} for r in reports]})
        # Exact chat-template tokenization includes complete prompts and completion
        # EOS; max_length=None disables budget rejection, never truncates history.
        pair=adapter.encode_pair(row)
        encoded.append(pair)
        sides.append({'relation':relation,'prompt_chars':len(row['prompt']),'prompt_tokens':len(pair.prompt_ids),
            'required_pair_tokens':pair.required_tokens,'ledger_events':len(state.receipts),
            'context_fact_count':len(source.context_ids),'raw_action_json':raw.as_json(),
            'raw_verdict':reports[0].immediate.verdict.value,'chosen_action':preference.chosen.kind,
            'request_reference_cost':reports[1].reference.best_success_cost,
            'prompt_token_ids_sha256':sha256(canonical(pair.prompt_ids).encode()).hexdigest(),
            'context_receipt':asdict(env.original_after_prefix.receipts[0]),
            'related_receipt_index':env.context_offset,'record_prompt_sha256':sha256(row['prompt'].encode()).hexdigest()})
        records.append(row)
    assert sides[0]['raw_action_json']==sides[1]['raw_action_json']
    assert sides[0]['ledger_events']==sides[1]['ledger_events']==spec['count']+1
    assert sides[0]['context_fact_count']==sides[1]['context_fact_count']
    assert sides[0]['prompt_chars']==sides[1]['prompt_chars']
    assert records[0]['chosen']==records[1]['rejected'] and records[0]['rejected']==records[1]['chosen']
    equal_tokens=sides[0]['prompt_tokens']==sides[1]['prompt_tokens']
    comparison={'equal_event_count':True,'equal_context_fact_count':True,'equal_complete_prompt_characters':True,
        'equal_complete_chat_prompt_tokens':equal_tokens,'actual_token_difference':sides[1]['prompt_tokens']-sides[0]['prompt_tokens'],
        'same_B_payload_action':True,'opposite_CI_verdict':True,'strict_preference_flip':True,
        'no_padding_truncation_or_text_compensation':True,'no_model_forward':True}
    for row in records:
        row['metadata']['matched_comparison']=comparison
        row['metadata']['tokenizer_provenance']=adapter.asset_provenance
        row['metadata'].pop('record_sha256',None)
        row['metadata']['record_sha256']=sha256(canonical(row).encode()).hexdigest()
    result={'spec':spec,'matched_pair_group_id':group,'records':records,'sides':sides,
            'comparison':comparison,'tokenizer':adapter.asset_provenance,'seconds':time.perf_counter()-started}
    write(output,result)
    print(json.dumps({'case_id':spec['case_id'],'equal_prompt_tokens':equal_tokens,'seconds':result['seconds']}))


def run(output):
    output.mkdir(parents=True,exist_ok=False);specs=specifications()
    write(output/'plan.json',{'protocol':PROTOCOL,'cases':specs,'hard_timeout_seconds':9.5,
        'tokenizer_manifest_sha256':ASSET_MANIFEST_SHA,'zero_length_placement_duplicates_excluded':True,
        'operations':'factory defaults; numeric threshold, set membership','variant':0,
        'long_agent_trajectory_generated':False,'workflow_groups':1,'primary_independent_holdout':False})
    snapshot=output/'source';snapshot.mkdir()
    for source in SOURCE_PATHS:
        destination=snapshot/source;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    receipts=[];results=[];raw=[]
    for spec in specs:
        path=output/'cases'/f"{spec['case_id']}.json";t=time.perf_counter()
        command=[sys.executable,'-B','-m','experiments.history_matched_probe','--worker',json.dumps(spec),'--output',str(path.resolve())]
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=9.5)
            receipt={'case_id':spec['case_id'],'returncode':process.returncode,'seconds':time.perf_counter()-t,
                     'stdout':process.stdout,'stderr':process.stderr,'hard_timeout_seconds':9.5}
        except subprocess.TimeoutExpired as error:
            receipt={'case_id':spec['case_id'],'timeout':True,'seconds':time.perf_counter()-t,
                     'stdout':str(error.stdout),'stderr':str(error.stderr),'hard_timeout_seconds':9.5}
        receipts.append(receipt);write(output/'worker_receipts.json',receipts)
        if receipt.get('returncode')!=0:raise RuntimeError('worker failed; raw evidence preserved')
        result=json.loads(path.read_text(encoding='utf-8'));results.append(result);raw.extend(result['records'])
    index=PreferenceCorpusIndex();unique={}
    for row in raw:
        index.add(row);unique.setdefault((row['prompt'],row['chosen'],row['rejected']),row)
    for name,rows in (('preferences_raw.jsonl',raw),('preferences_unique.jsonl',list(unique.values()))):
        (output/name).write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
    (output/'matched_pairs.jsonl').write_text(''.join(canonical({'matched_pair_group_id':r['matched_pair_group_id'],
        'spec':r['spec'],'sides':r['sides'],'comparison':r['comparison']})+'\n' for r in results),encoding='utf-8')
    summary={'protocol':PROTOCOL,'cases':len(specs),'successful_workers':len(receipts),
        'raw_preferences':len(raw),'unique_preferences':len(unique),
        'positive_history_unique_preferences':sum(r['metadata']['synthetic_background_receipts']>0 for r in unique.values()),
        'equal_events_and_characters_pairs':sum(r['comparison']['equal_event_count'] and r['comparison']['equal_complete_prompt_characters'] for r in results),
        'equal_full_chat_prompt_token_pairs':sum(r['comparison']['equal_complete_chat_prompt_tokens'] for r in results),
        'non_equal_token_pairs':[{'case_id':r['spec']['case_id'],'difference':r['comparison']['actual_token_difference']}
            for r in results if not r['comparison']['equal_complete_chat_prompt_tokens']],
        'strict_preference_flip_pairs':sum(r['comparison']['strict_preference_flip'] for r in results),
        'complete_consequence_replays':len(raw)*2,'max_worker_seconds':max(r['seconds'] for r in receipts),
        'hard_timeout_seconds':9.5,'prompt_tokens_min':min(s['prompt_tokens'] for r in results for s in r['sides']),
        'prompt_tokens_max':max(s['prompt_tokens'] for r in results for s in r['sides']),
        'required_pair_tokens_max':max(s['required_pair_tokens'] for r in results for s in r['sides']),
        'tokenizer':results[0]['tokenizer'],'corpus_index':index.summary(),
        'source_sha256':raw[0]['metadata']['source_sha256'],
        'output_sha256':{n:sha256((output/n).read_bytes()).hexdigest() for n in
            ('preferences_raw.jsonl','preferences_unique.jsonl','matched_pairs.jsonl')},
        'human_verified':False,'model_forward_or_training':False,'long_agent_trajectory_generated':False,
        'primary_independent_holdout':False,'scope':'equal-length semantic-context control inside one synthetic shared workflow'}
    write(output/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.worker:worker(json.loads(args.worker),Path(args.output))
    else:run(Path(args.output))
