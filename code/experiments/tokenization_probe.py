"""CPU Qwen tokenizer/data invariants. No model forward, backward, or training."""
from __future__ import annotations
import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/20261112_cpu_pilot/tokenization'
LIMITS=[256,512,1024,2048,4096]
TIMEOUT=9.5


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def digest(value):return sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def assets():
    sys.path.insert(0,str(OUT/'dependencies'))
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    cfg=json.loads((OUT/'official_tokenizer/tokenizer_config.json').read_text(encoding='utf-8'))
    tokenizer=Tokenizer.from_file(str(OUT/'official_tokenizer/tokenizer.json'))
    template=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True).from_string(cfg['chat_template'])
    def render(prompt,answer=None):
        messages=[{'role':'user','content':prompt}]
        if answer is not None:messages.append({'role':'assistant','content':answer})
        return template.render(messages=messages,tools=None,add_generation_prompt=answer is None)
    return tokenizer,render,tokenizer.token_to_id(cfg['eos_token']),tokenizer.token_to_id(cfg['pad_token'])


def encode_pair(tokenizer,render,eos,row):
    prompt=tokenizer.encode(render(row['prompt']),add_special_tokens=False).ids
    result={'row_index':row.get('row_index'),'prompt_ids':prompt,'prompt_sha256':sha256(row['prompt'].encode()).hexdigest()}
    for side in ('chosen','rejected'):
        ids=tokenizer.encode(render(row['prompt'],row[side]),add_special_tokens=False).ids
        if ids[:len(prompt)]!=prompt:raise ValueError('chat-template tokenized prefix differs: '+side)
        stop=[i for i in range(len(prompt),len(ids)) if ids[i]==eos]
        if len(stop)!=1:raise ValueError('completion requires exactly one unambiguous EOS: '+side)
        end=stop[0]+1
        result[side]={'completion_ids':ids[len(prompt):end],'template_tail_ids':ids[end:],
                      'total_input_tokens':len(ids),'supervised_tokens':end-len(prompt)}
        if result[side]['supervised_tokens']<2:raise ValueError('completion content was lost')
    return result


def tokenize_chunk(chunk):
    tokenizer,render,eos,pad=assets()
    rows=[json.loads(line) for line in (OUT/'input_chunks'/f'{chunk:03d}.jsonl').read_text(encoding='utf-8').splitlines()]
    encoded=[];failures=[]
    truncation={str(n):{'pair_fully_retained':0,'pair_rejected':0,'naive_right_zero_supervision_branches':0,
                        'naive_right_loses_eos_branches':0} for n in LIMITS}
    for row in rows:
        try:pair=encode_pair(tokenizer,render,eos,row)
        except ValueError as error:
            failures.append({'row_index':row['row_index'],'error':str(error)});continue
        encoded.append(pair);p=len(pair['prompt_ids'])
        for n in LIMITS:
            retained=all(pair[s]['total_input_tokens']<=n for s in ('chosen','rejected'))
            truncation[str(n)]['pair_fully_retained' if retained else 'pair_rejected']+=1
            for side in ('chosen','rejected'):
                truncation[str(n)]['naive_right_zero_supervision_branches']+=int(n<=p)
                truncation[str(n)]['naive_right_loses_eos_branches']+=int(n<p+pair[side]['supervised_tokens'])
    target=OUT/'encoded_chunks'/f'{chunk:03d}.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in encoded),encoding='utf-8')
    result={'chunk':chunk,'input_rows':len(rows),'verified_shared_prompt_pairs':len(encoded),
            'eos_supervised_branches':2*len(encoded),'failures':failures,'truncation':truncation,
            'prompt_token_min':min((len(r['prompt_ids']) for r in encoded),default=None),
            'prompt_token_max':max((len(r['prompt_ids']) for r in encoded),default=None),
            'masked_template_tail_tokens':sum(len(r[s]['template_tail_ids']) for r in encoded for s in ('chosen','rejected')),
            'eos_token_id':eos,'pad_token_id':pad,'encoded_sha256':sha256(target.read_bytes()).hexdigest()}
    dump(OUT/'results'/f'chunk_{chunk:03d}.json',result)


def padding_probe():
    import numpy as np
    _,_,eos,pad=assets()
    rows=[]
    for file in sorted((OUT/'encoded_chunks').glob('*.jsonl')):
        rows.extend(json.loads(line) for line in file.read_text(encoding='utf-8').splitlines())
    # Spread selections across the frozen corpus, including unequal prompt/completion lengths.
    chosen_rows=[rows[i*(len(rows)-1)//31] for i in range(32)]
    sequences=[]
    for pair in chosen_rows:
        for side in ('chosen','rejected'):
            p=pair['prompt_ids'];c=pair[side]['completion_ids'];tail=pair[side]['template_tail_ids']
            ids=p+c+tail;labels=[-100]*len(p)+c+[-100]*len(tail)
            target=np.asarray(ids[1:]);active=np.asarray(labels[1:])!=-100
            logp=-.01*(1+target%17)-1e-6*np.arange(len(ids)-1)
            sequences.append((ids,labels,float(logp[active].sum()),float(logp[active].mean())))
    checks=[];all_max_error=0.0
    for pad_id in (pad,eos):
        for side in ('left','right'):
            for batch_size in (1,2,8,32):
                verified=0;max_error=0.0;eos_kept=0;buggy_eos_drops=0
                for begin in range(0,len(sequences),batch_size):
                    batch=sequences[begin:begin+batch_size];width=max(len(x[0]) for x in batch)
                    ids=np.full((len(batch),width),pad_id,dtype=np.int64)
                    labels=np.full_like(ids,-100);attention=np.zeros_like(ids)
                    for i,(tokens,labs,_,_) in enumerate(batch):
                        offset=width-len(tokens) if side=='left' else 0
                        ids[i,offset:offset+len(tokens)]=tokens;labels[i,offset:offset+len(tokens)]=labs
                        attention[i,offset:offset+len(tokens)]=1
                    positions=np.maximum(attention.cumsum(axis=1)-1,0)
                    target=ids[:,1:];active=(labels[:,1:]!=-100)&(attention[:,1:]==1)
                    lp=-.01*(1+target%17)-1e-6*positions[:,:-1]
                    for i,(tokens,labs,expected_sum,expected_mean) in enumerate(batch):
                        assert np.array_equal(positions[i,attention[i]==1],np.arange(len(tokens)))
                        assert np.array_equal(labels[i,attention[i]==1],np.asarray(labs))
                        assert int(((ids[i]==eos)&(labels[i]!=-100)).sum())==1
                        got_sum=float(lp[i,active[i]].sum());got_mean=float(lp[i,active[i]].mean())
                        error=max(abs(got_sum-expected_sum),abs(got_mean-expected_mean));max_error=max(max_error,error)
                        assert error<1e-12
                        eos_kept+=1;verified+=1
                        buggy_eos_drops+=int(((ids[i]==eos)&(labels[i]!=-100)).sum()) if pad_id==eos else 0
                checks.append({'padding':side,'batch_size':batch_size,'pad_id':pad_id,'sequences_verified':verified,
                               'supervised_eos_preserved':eos_kept,'max_abs_error':max_error,
                               'naive_mask_by_pad_token_would_drop_eos':buggy_eos_drops})
                all_max_error=max(all_max_error,max_error)
    dump(OUT/'results/padding.json',{'classification':'synthetic target-logprob reduction, no model forward',
         'unique_sequences':len(sequences),'checks':checks,'total_sequence_checks':sum(x['sequences_verified'] for x in checks),
         'max_abs_error':all_max_error,'position_id_rule':'max(cumsum(attention_mask)-1,0)',
         'alignment':'logits positions[:-1] score labels[1:]; prompt/pad/template-tail labels=-100'})


def boundary_probe():
    tokenizer,render,eos,pad=assets()
    original=[json.loads(x) for x in (OUT/'input_snapshot.jsonl').read_text(encoding='utf-8').splitlines()]
    row=next(r for r in original if json.loads(r['prompt']).get('authorized_delivered_messages'))
    observation=json.loads(row['prompt'])
    critical=next(m['body'] for m in observation['authorized_delivered_messages'] if m.get('body'))
    observation['task']+='\n'+'Irrelevant historical note. '*3000
    stress=dict(row,prompt=json.dumps(observation,ensure_ascii=False,sort_keys=True))
    pair=encode_pair(tokenizer,render,eos,stress);p=pair['prompt_ids']
    long_results=[]
    for n in LIMITS:
        ids=p+pair['chosen']['completion_ids']+pair['chosen']['template_tail_ids']
        retained=len(ids)<=n
        left_decoded=tokenizer.decode(ids[-n:],skip_special_tokens=False)
        long_results.append({'max_length':n,'full_sequence_tokens':len(ids),'strict_pair_retained':retained,
                             'naive_left_truncation_loses_critical_receipt_body':critical not in left_decoded,
                             'naive_right_truncation_zero_supervision':n<=len(p)})
        assert not retained and critical not in left_decoded and n<=len(p)
    contaminated=dict(row,chosen=json.dumps({'body':'literal '+ '<|im_end|>'},sort_keys=True))
    try:encode_pair(tokenizer,render,eos,contaminated);blocked=False
    except ValueError:blocked=True
    assert blocked
    # A valid prefix boundary is checked on complete template renders, never by
    # assuming separate prompt/completion tokenizations concatenate identically.
    dump(OUT/'results/boundary.json',{'long_history_is_control_only':True,
         'critical_receipt_body_sha256':sha256(critical.encode()).hexdigest(),'long_history_checks':long_results,
         'embedded_special_eos_completion_blocked':blocked,'template_used_verbatim':True,
         'policy':'Never export a truncated pair if any prompt/history, completion, or terminal EOS is removed.'})


def objective_probe():
    # Token-level policy/reference log-ratio differences; no trainable model.
    chosen=[.1,.1];rejected=[.15]
    summed=sum(chosen)-sum(rejected);averaged=sum(chosen)/len(chosen)-sum(rejected)/len(rejected)
    assert summed>0 and averaged<0
    dump(OUT/'results/objective.json',{'classification':'synthetic log-ratio arithmetic counterexample',
         'chosen_token_logratios':chosen,'rejected_token_logratios':rejected,
         'sequence_sum_margin':summed,'sequence_mean_margin':averaged,'ranking_reverses':True,
         'implementation_policy':'Completion conditional log-likelihood sums remain the declared DPO quantity; mean is diagnostic only.',
         'real_model_forward':False,'real_model_backward':False})


def run():
    for name in ('input_chunks','encoded_chunks','results','worker_logs'):(OUT/name).mkdir(exist_ok=True)
    source=ROOT/'artifacts/20261112_cpu_pilot/dataset/preferences.jsonl'
    raw=source.read_bytes();(OUT/'input_snapshot.jsonl').write_bytes(raw)
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    chunks=[]
    for start in range(0,len(rows),128):
        index=start//128;path=OUT/'input_chunks'/f'{index:03d}.jsonl'
        selected=[{'row_index':i,'prompt':rows[i]['prompt'],'chosen':rows[i]['chosen'],'rejected':rows[i]['rejected'],
                   'record_sha256':rows[i]['metadata'].get('record_sha256')} for i in range(start,min(start+128,len(rows)))]
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8');chunks.append(index)
    metadata=json.loads((OUT/'official_model_metadata.json').read_text(encoding='utf-8'))
    source_bytes=Path(__file__).read_bytes();(OUT/'source_snapshot.py').write_bytes(source_bytes)
    config={'input_path':str(source),'input_sha256':sha256(raw).hexdigest(),'input_rows':len(rows),
            'source_sha256':sha256(source_bytes).hexdigest(),'official_repo':'Qwen/Qwen2.5-7B-Instruct',
            'official_revision':metadata['sha'],'maximum_lengths':LIMITS,'chunk_size':128,'worker_timeout_seconds':TIMEOUT,
            'tokenizer_files':{p.name:sha256(p.read_bytes()).hexdigest() for p in (OUT/'official_tokenizer').glob('*.json')},
            'wheels':{p.name:{'sha256':sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in (OUT/'wheels').glob('*.whl')},
            'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(OUT/'dependencies')])},
            'runtime_python':sys.version,'scope':'CPU tokenizer/preprocessing and synthetic reduction only; NO model forward/backward'}
    dump(OUT/'config.json',config)
    tasks=[('tokenize',i) for i in chunks]+[('padding',None),('boundary',None),('objective',None)]
    receipts=[]
    for unit,chunk in tasks:
        label=unit+(f'_{chunk:03d}' if chunk is not None else '')
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',unit]
        if chunk is not None:command+=['--chunk',str(chunk)]
        begin=time.perf_counter()
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=TIMEOUT)
            receipt={'unit':label,'returncode':result.returncode,'elapsed_seconds':time.perf_counter()-begin,'timeout_seconds':TIMEOUT}
            (OUT/'worker_logs'/(label+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        except subprocess.TimeoutExpired:
            receipt={'unit':label,'returncode':None,'timed_out':True,'elapsed_seconds':time.perf_counter()-begin,'timeout_seconds':TIMEOUT}
        receipts.append(receipt);print(json.dumps(receipt),flush=True)
    dump(OUT/'worker_receipts.json',receipts)
    outputs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'results').glob('chunk_*.json'))]
    failed=[r for r in receipts if r['returncode']!=0]
    aggregate={str(n):{k:sum(r['truncation'][str(n)][k] for r in outputs) for k in
                ('pair_fully_retained','pair_rejected','naive_right_zero_supervision_branches','naive_right_loses_eos_branches')} for n in LIMITS}
    summary={'status':'PASS' if not failed and sum(len(r['failures']) for r in outputs)==0 else 'INCOMPLETE_OR_FAILED',
             'input_rows':len(rows),'input_sha256':config['input_sha256'],'source_sha256':config['source_sha256'],
             'shared_prompt_verified_pairs':sum(r['verified_shared_prompt_pairs'] for r in outputs),
             'supervised_eos_branches':sum(r['eos_supervised_branches'] for r in outputs),
             'masked_template_tail_tokens':sum(r['masked_template_tail_tokens'] for r in outputs),
             'row_failures':[x for r in outputs for x in r['failures']],'failed_workers':failed,
             'prompt_token_range':[min(r['prompt_token_min'] for r in outputs),max(r['prompt_token_max'] for r in outputs)],
             'truncation':aggregate,'workers':len(receipts),'max_worker_seconds':max(r['elapsed_seconds'] for r in receipts),
             'official_revision':metadata['sha'],'real_model_forward':False,'real_model_backward':False,
             'gpu_numeric_validation':False,'synthetic_logprob_only':True}
    for name in ('padding','boundary','objective'):
        path=OUT/'results'/(name+'.json')
        if path.exists():summary[name]=json.loads(path.read_text(encoding='utf-8'))
    dump(OUT/'summary.json',summary)
    print('SUMMARY='+str(OUT/'summary.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['tokenize','padding','boundary','objective'])
    parser.add_argument('--chunk',type=int);args=parser.parse_args()
    if args.worker=='tokenize':tokenize_chunk(args.chunk)
    elif args.worker=='padding':padding_probe()
    elif args.worker=='boundary':boundary_probe()
    elif args.worker=='objective':objective_probe()
    else:run()


if __name__=='__main__':main()
