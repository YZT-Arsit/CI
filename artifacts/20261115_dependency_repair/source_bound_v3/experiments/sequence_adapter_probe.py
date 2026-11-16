"""Bounded CPU validation/export of complete pinned-Qwen preference sequences."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from kci.training_data import (EncodedPreference,EncodedSequence,IGNORE_INDEX,PinnedTokenizerAssets,
    PreferenceCorpusIndex,TokenBudgetExceeded,TrainingDataError,TrainingSequenceAdapter,canonical)

BUDGETS=(256,512,1024,2048,4096,8192,16384,32768,65536)
HISTORY_COUNTS=(0,8,32,128,256,512)
TIMEOUT=9.2


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_lines(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def adapter(config):
    assets=PinnedTokenizerAssets.from_manifest(config['asset_manifest'],expected_manifest_sha256=config['asset_manifest_sha256'])
    return TrainingSequenceAdapter(assets)


def reconstruct(value):
    def seq(item):return EncodedSequence(**{k:tuple(v) if isinstance(v,list) else v for k,v in item.items()})
    return EncodedPreference(tuple(value['prompt_ids']),seq(value['chosen']),seq(value['rejected']),value['provenance'])


def row_statistics(pair,record,row_index):
    obs=json.loads(record['prompt']);meta=record['metadata']
    return {'row_index':row_index,'record_id':meta.get('record_id'),'required_tokens':pair.required_tokens,
            'prompt_tokens':len(pair.prompt_ids),'chosen_supervised_tokens':pair.chosen.completion_end-len(pair.prompt_ids),
            'rejected_supervised_tokens':pair.rejected.completion_end-len(pair.prompt_ids),
            'history_events':len(obs['authorized_delivered_messages']),'role':obs['role'],
            'base_group':meta.get('base_case_group_id',meta.get('base_group_id',meta['scenario_group_id'])),
            'rule_family_id':meta['rule_family_id'],'split':meta.get('split','unassigned'),
            'source_evidence_class':meta.get('evidence_class'),'training_eligible':pair.provenance['training_eligible']}


def encode_chunk(root,config,index):
    encode=adapter(config);rows=load_lines(root/'input_chunks'/f'{index:03d}.jsonl')
    stats=[];failures=[];canary_checks=0
    output=root/'encoded_chunks'/f'{index:03d}.jsonl';output.parent.mkdir(exist_ok=True)
    with output.open('w',encoding='utf-8') as handle:
        for item in rows:
            record=item['record'];position=item['row_index']
            try:
                pair=encode.encode_pair(record)
                polluted=deepcopy(record)
                polluted['metadata']['reference']='HIDDEN_REFERENCE_CANARY_91'
                polluted['metadata']['witness']='HIDDEN_WITNESS_CANARY_92'
                polluted['private_metadata']='HIDDEN_METADATA_CANARY_93'
                twin=encode.encode_pair(polluted)
                assert (pair.prompt_ids,pair.chosen,pair.rejected)==(twin.prompt_ids,twin.chosen,twin.rejected)
                canary_checks+=1
                handle.write(canonical({'row_index':position,**pair.as_dict()})+'\n')
                stats.append(row_statistics(pair,record,position))
            except TrainingDataError as error:failures.append({'row_index':position,'reason':str(error)})
    save(root/'results'/f'chunk_{index:03d}.json',{'input_rows':len(rows),'encoded_rows':len(stats),
         'metadata_canary_noninterference_checks':canary_checks,'stats':stats,'failures':failures,
         'output_sha256':sha256(output.read_bytes()).hexdigest(),'assets':encode.asset_provenance})


def long_control(root,config,index):
    encode=adapter(config);item=json.loads((root/'long_inputs'/f'{index:03d}.json').read_text(encoding='utf-8'))
    record=item['record'];pair=encode.encode_pair(record,structural_control=True)
    decoded=encode.tokenizer.decode(list(pair.chosen.input_ids),skip_special_tokens=False)
    assert record['prompt'] in decoded and item['critical_body'] in decoded
    checks=[]
    for budget in BUDGETS:
        try:
            candidate=encode.encode_pair(record,max_length=budget,structural_control=True)
            assert candidate==pair;retained=True
        except TokenBudgetExceeded as error:
            assert error.required_tokens==pair.required_tokens;retained=False
        clipped=encode.tokenizer.decode(list(pair.chosen.input_ids[-budget:]),skip_special_tokens=False)
        checks.append({'max_length':budget,'complete_retained':retained,
                       'naive_left_clip_loses_critical_body':item['critical_body'] not in clipped,
                       'naive_right_clip_loses_all_supervision':budget<=len(pair.prompt_ids)})
    padding_checks=0
    for side in ('left','right'):
        batch=encode.collate([pair],padding_side=side,pad_to_length=pair.required_tokens+3,
                            pad_token_id=encode.eos_token_id,allow_structural_controls=True)
        for branch in ('chosen','rejected'):
            ids=batch[branch]['input_ids'][0];mask=batch[branch]['attention_mask'][0]
            assert tuple(v for v,m in zip(ids,mask) if m)==getattr(pair,branch).input_ids
            assert batch[branch]['labels'][0].count(encode.eos_token_id)==1;padding_checks+=1
    try:encode.collate([pair]);raise AssertionError('unlabeled controls entered a training batch')
    except TrainingDataError:pass
    save(root/'long_encoded'/f'{index:03d}.json',pair.as_dict())
    save(root/'results'/f'long_{index:03d}.json',{'control_id':item['control_id'],
        'benign_history_count':item['benign_history_count'],'critical_location':item['critical_location'],
        'required_tokens':pair.required_tokens,'prompt_tokens':len(pair.prompt_ids),
        'actual_history_events':len(json.loads(record['prompt'])['authorized_delivered_messages']),
        'full_prompt_and_critical_body_preserved':True,'training_eligible':False,
        'padding_checks':padding_checks,'budgets':checks,
        'scope':'unlabeled sequence-only control; new histories are not certified CI/preference labels'})


def padding_probe(root,config):
    encode=adapter(config);values=[]
    for path in sorted((root/'encoded_chunks').glob('*.jsonl')):
        part=load_lines(path)
        if part:values.extend((part[0],part[-1]))
    pairs=[reconstruct(v) for v in values]
    checks=0;max_error=0.
    for side in ('left','right'):
        for pad in (encode.pad_token_id,encode.eos_token_id):
            batch=encode.collate(pairs,padding_side=side,pad_token_id=pad)
            for branch in ('chosen','rejected'):
                for i,pair in enumerate(pairs):
                    sequence=getattr(pair,branch);ids=batch[branch]['input_ids'][i]
                    labs=batch[branch]['labels'][i];attention=batch[branch]['attention_mask'][i];positions=batch[branch]['position_ids'][i]
                    assert [v for v,m in zip(positions,attention) if m]==list(range(len(sequence.input_ids)))
                    assert [v for v,m in zip(labs,attention) if m]==list(sequence.labels)
                    # Synthetic target log probabilities depend on actual causal position, never physical padded offset.
                    expected=sum(-.01*(1+sequence.input_ids[t]%17)-1e-6*(t-1)
                                 for t in range(1,len(sequence.input_ids)) if sequence.labels[t]!=IGNORE_INDEX)
                    observed=sum(-.01*(1+ids[t]%17)-1e-6*positions[t-1]
                                 for t in range(1,len(ids)) if labs[t]!=IGNORE_INDEX and attention[t])
                    error=abs(expected-observed);assert error<1e-12;max_error=max(max_error,error)
                    assert labs.count(encode.eos_token_id)==1;checks+=1
    save(root/'results/padding.json',{'pairs':len(pairs),'sequence_padding_checks':checks,'max_reduction_error':max_error,
        'scope':'synthetic causal logprob reduction; no model forward/backward'})


def budget_stats(rows):
    return {str(b):{'retained':sum(r['required_tokens']<=b for r in rows),
                    'rejected_without_truncation':sum(r['required_tokens']>b for r in rows),
                    'retention_rate':sum(r['required_tokens']<=b for r in rows)/len(rows) if rows else None} for b in BUDGETS}


def aggregate(root,config,receipts):
    chunks=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'results').glob('chunk_*.json'))]
    stats=[r for c in chunks for r in c['stats']];failures=[r for c in chunks for r in c['failures']]
    grouped={}
    for field in ('base_group','rule_family_id','split','role','source_evidence_class'):
        grouped[field]={str(key):{'rows':len(subset),'budgets':budget_stats(subset)}
            for key in sorted(set(r[field] for r in stats),key=str)
            for subset in [[r for r in stats if r[field]==key]]}
    tokens=sorted(r['required_tokens'] for r in stats)
    quantiles={str(q):tokens[round(q*(len(tokens)-1))] if tokens else None for q in (0,.25,.5,.75,.9,.95,.99,1)}
    long=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'results').glob('long_*.json'))]
    result={'input_rows':config['input_rows'],'fully_encoded_rows':len(stats),'row_failures':failures,
            'schema_canary_checks':sum(c['metadata_canary_noninterference_checks'] for c in chunks),
            'budget_summary':budget_stats(stats),'grouped_budget_summary':grouped,'token_quantiles':quantiles,
            'prompt_tokens_range':[min(r['prompt_tokens'] for r in stats),max(r['prompt_tokens'] for r in stats)] if stats else [],
            'actual_training_history_event_range':[min(r['history_events'] for r in stats),max(r['history_events'] for r in stats)] if stats else [],
            'controlled_long_sequences':long,'long_controls_counted_as_training_rows':False,
            'corpus_index':config['corpus_index'],'workers':len(receipts),
            'failed_workers':[r for r in receipts if r['status']!='completed'],
            'max_worker_seconds':max(r['seconds'] for r in receipts),
            'scope':'reusable CPU sequence contract; not HF Trainer, semantic label verification, or model training'}
    padding=root/'results/padding.json'
    if padding.exists():result['padding']=json.loads(padding.read_text(encoding='utf-8'))
    result['status']='PASS' if not failures and not result['failed_workers'] and len(stats)==config['input_rows'] else 'INCOMPLETE_OR_FAILED'
    save(root/'summary.json',result)
    print(json.dumps({'status':result['status'],'rows':len(stats),'long_controls':len(long),'max_worker_seconds':result['max_worker_seconds']}))


def run(dataset,asset_manifest,output,chunk_size,skip_long_controls=False):
    root=Path(output).resolve()
    if (root/'manifest.json').exists():raise FileExistsError('Output manifest already exists; do not overwrite a frozen run')
    for folder in ('input_chunks','encoded_chunks','long_inputs','long_encoded','results','logs'):(root/folder).mkdir(parents=True,exist_ok=True)
    raw=Path(dataset).read_bytes();rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    index=PreferenceCorpusIndex()
    for row in rows:index.add(row)
    chunks=[]
    for start in range(0,len(rows),chunk_size):
        number=start//chunk_size;chunks.append(number)
        (root/'input_chunks'/f'{number:03d}.jsonl').write_text(''.join(canonical({'row_index':i,'record':rows[i]})+'\n'
            for i in range(start,min(start+chunk_size,len(rows)))),encoding='utf-8')
    long_inputs=[]
    if not skip_long_controls:
        base=next(r for r in rows if json.loads(r['prompt'])['role']=='B' and any('identity' in m['fact_ids'] for m in json.loads(r['prompt'])['authorized_delivered_messages']))
        observation=json.loads(base['prompt']);critical=next(m for m in observation['authorized_delivered_messages'] if 'identity' in m['fact_ids'])
        time_fact=next(f for f in observation['local_facts'] if f['id']=='time')
        for count in HISTORY_COUNTS:
            for location in ('first','last'):
                number=len(long_inputs);control=deepcopy(base);obs=deepcopy(observation)
                benign=[{'event_id':f'control_{j:04d}','sender':'B','recipient':'R','purpose':obs['purpose'],
                         'fact_ids':['time'],'body':time_fact['text'],'delivered':True} for j in range(count)]
                obs['authorized_delivered_messages']=([critical]+benign if location=='first' else benign+[critical])
                control['prompt']=canonical(obs)
                control['metadata'].update(certified=False,control_only=True,evidence_class='synthetic_sequence_control',
                                           record_id=f'long_sequence_control_{number:03d}',split='unassigned')
                item={'control_id':control['metadata']['record_id'],'benign_history_count':count,'critical_location':location,
                      'critical_body':critical['body'],'record':control}
                save(root/'long_inputs'/f'{number:03d}.json',item);long_inputs.append(number)
    source=Path(__file__);module=Path(__file__).resolve().parents[1]/'kci/training_data.py'
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_path':str(Path(dataset).resolve()),
            'input_sha256':sha256(raw).hexdigest(),'input_rows':len(rows),'asset_manifest':str(Path(asset_manifest).resolve()),
            'asset_manifest_sha256':sha256(Path(asset_manifest).read_bytes()).hexdigest(),'chunk_size':chunk_size,
            'budgets':BUDGETS,'long_history_counts':HISTORY_COUNTS,'long_controls':len(long_inputs),'skip_legacy_long_controls':skip_long_controls,'worker_timeout_seconds':TIMEOUT,
            'source_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in (source,module)},'corpus_index':index.summary(),
            'long_control_policy':'uncertified structural-only sequences, excluded from preference corpus and training collator'}
    save(root/'manifest.json',config)
    (root/'probe_source_snapshot.py').write_bytes(source.read_bytes());(root/'adapter_source_snapshot.py').write_bytes(module.read_bytes())
    jobs=[('encode',i) for i in chunks]+[('long',i) for i in long_inputs]+[('padding',None)]
    receipts=[]
    for kind,number in jobs:
        command=[sys.executable,'-B','-m','experiments.sequence_adapter_probe','--worker',kind,'--output',str(root)]
        if number is not None:command+=['--index',str(number)]
        start=time.perf_counter()
        try:
            process=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=TIMEOUT)
            status='completed' if process.returncode==0 else 'failed';code=process.returncode
            stdout,stderr=process.stdout,process.stderr
        except subprocess.TimeoutExpired as error:status='timeout';code=None;stdout=str(error.stdout or '');stderr=str(error.stderr or '')
        receipt={'worker':kind,'index':number,'seconds':time.perf_counter()-start,'status':status,'returncode':code,'timeout':TIMEOUT}
        receipts.append(receipt);save(root/'logs'/f'{kind}_{number}.json',dict(receipt,stdout=stdout,stderr=stderr))
        print(json.dumps(receipt),flush=True)
    save(root/'worker_receipts.json',receipts);aggregate(root,config,receipts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset');parser.add_argument('--asset-manifest')
    parser.add_argument('--output',required=True);parser.add_argument('--chunk-size',type=int,default=96)
    parser.add_argument('--skip-long-controls',action='store_true',help='encode a different corpus without generating legacy toy-specific controls')
    parser.add_argument('--worker',choices=('encode','long','padding'));parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.worker:
        root=Path(args.output);config=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if args.worker=='encode':encode_chunk(root,config,args.index)
        elif args.worker=='long':long_control(root,config,args.index)
        else:padding_probe(root,config)
    else:run(args.dataset,args.asset_manifest,args.output,args.chunk_size,args.skip_long_controls)
