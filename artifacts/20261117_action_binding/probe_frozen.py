"""Frozen teacher-prefix interventions. Scores are diagnostics, not policy success."""
import sys,json,hashlib,time,re,random,threading,os,gc
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];OLD=ROOT/'artifacts/20261116_relation_solution';sys.path.insert(0,str(OLD))
from lm_common import setup,palette,answer_ids,torch,prompt,seed_for
from proof_targets import response
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

def inputs_for(tokenizer,row,other,pool,arm,mode,nonce):
    text,actions=prompt(row['observation'],pool,nonce)
    text+=('\nGive each option SAFE (immediate), TASK (safe completion possible), PATH (may include mandatory future links), then ANSWER:\nletter.' if arm=='trace' else '\nReply ANSWER:\nletter.')
    base=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
    evidence=other if mode=='swapped_trace' else row
    prefix='ANSWER:\n'
    if mode!='direct':
        target,_=response(evidence,actions,pool,nonce,'trace',random.Random(0))
        prefix=target.split('ANSWER:')[0]+'ANSWER:\n'
    tok=tokenizer(prefix,add_special_tokens=False,return_offsets_mapping=True);bits=[]
    for match in re.finditer(r'(SAFE|TASK) ([01])',prefix):
        begin,end=match.span(2);indices=[i for i,(a,b) in enumerate(tok['offset_mapping']) if a<end and b>begin]
        assert len(indices)==1
        bits.append({'kind':match[1],'value':int(match[2]),'index':indices[0]})
    ids=base+tok['input_ids'];assert len(ids)<=1024
    return {'ids':ids,'base_length':len(base),'suffix_length':len(tok['input_ids']),'bits':bits,'prompt':text,'prefix':prefix,'actions':actions}

def infer(model,tokenizer,items):
    def timeout():
        (RUN/'TIMEOUT.json').write_text(json.dumps({'stage':'forward','seconds':9.5}));os._exit(124)
    timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter()
    length=max(len(x['ids']) for x in items);keep=max(x['suffix_length'] for x in items)+1
    ids=torch.tensor([[tokenizer.pad_token_id]*(length-len(x['ids']))+x['ids'] for x in items]);mask=torch.tensor([[0]*(length-len(x['ids']))+[1]*len(x['ids']) for x in items])
    with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask,logits_to_keep=keep,use_cache=False).logits
    elapsed=time.perf_counter()-start;timer.cancel();assert elapsed<9.5
    return logits,length-keep,elapsed

def run():
    plan=json.loads((RUN/'PLAN.json').read_text(encoding='utf-8'));assert hashlib.sha256((RUN/'cases.jsonl').read_bytes()).hexdigest()==plan['cases_sha256']
    assert not (RUN/'PROBE_PLAN.json').exists()
    source_files=[Path(__file__),OLD/'lm_common.py',OLD/'proof_targets.py',ROOT/'code/kci/relation_language.py']
    source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (RUN/'PROBE_PLAN.json').write_text(json.dumps({'source_hashes':source_hashes,'batch':2,'max_forward_seconds':9.5,'model_selection':plan['models'],'modes':plan['modes'],'no_training':True,'no_free_generation_claim':True},indent=2))
    rows=list(map(json.loads,(RUN/'cases.jsonl').read_text(encoding='utf-8').splitlines()));groups={}
    for row in rows:groups.setdefault(row['pair_id'],[]).append(row)
    controls=[];reports=[]
    for name in plan['models']:
        start=time.perf_counter();tokenizer,model=setup(17);arm='plain'
        if name=='base':model.disable_adapter_layers();checkpoint_hashes={}
        else:
            parent='balanced_language_models' if name.startswith('balanced') else 'language_models';folder=OLD/parent/name
            sealed=json.loads((folder/'sealed.json').read_text());checkpoint_hashes=sealed['adapter_hashes'];arm=sealed['arm']
            assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in checkpoint_hashes.items())
            set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')))
        model.eval();load_seconds=time.perf_counter()-start;pools=palette(tokenizer);letters=answer_ids(tokenizer);binary=[tokenizer.encode(str(i),add_special_tokens=False)[0] for i in (0,1)]
        first=rows[0];nonce=seed_for(first['pair_id'],plan['nonce']);item=inputs_for(tokenizer,first,rows[1],pools['train'],arm,'direct',nonce)
        identity,_,seconds=infer(model,tokenizer,[item,item]);difference=float((identity[0,-1].softmax(-1)-identity[1,-1].softmax(-1)).abs().max());assert difference<1e-6
        controls.append({'model':name,'identity_max_probability_difference':difference,'seconds':seconds});records=[]
        for mode in plan['modes']:
            for pair_id,pair in groups.items():
                assert [r['side'] for r in pair]==[0,1];nonce=seed_for(pair_id,plan['nonce']);pool=pools['train' if pair[0]['split']=='train' else 'fresh']
                items=[inputs_for(tokenizer,r,pair[1-i],pool,arm,mode,nonce) for i,r in enumerate(pair)]
                assert items[0]['actions']==items[1]['actions']
                assert len(items[0]['ids'])==len(items[1]['ids']),('pair length mismatch',pair_id,mode)
                logits,offset,seconds=infer(model,tokenizer,items)
                for i,(row,item) in enumerate(zip(pair,items)):
                    last=logits[i,-1];p=last.softmax(-1);allowed=letters[:len(item['actions'])];top=int(last.argmax());selected=item['actions'][allowed.tolist().index(top)] if top in allowed.tolist() else None
                    bit_scores=[]
                    for bit in item['bits']:
                        position=item['base_length']+bit['index']-1-offset
                        assert bit['value'] in (0,1) and item['ids'][item['base_length']+bit['index']]==binary[bit['value']]
                        scores=logits[i,position];pred=int(scores.argmax())
                        bit_scores.append({'kind':bit['kind'],'target':bit['value'],'binary_p1':float(scores[binary].softmax(0)[1]),'full_vocab_correct':pred==binary[bit['value']]})
                    record={'model':name,'mode':mode,'pair_id':pair_id,'side':row['side'],'split':row['split'],'depth':row['depth'],'selected':selected,'optimal_action':row['optimal_action'],'actions':item['actions'],'legal_argmax_action':item['actions'][int(last[allowed].argmax())],'legal_probabilities':p[allowed].tolist(),'legal_mass':float(p[allowed].sum()),'full_vocab_top_token':tokenizer.decode([top]),'full_vocab_top_token_id':top,'token_count':len(item['ids']),'prompt':item['prompt'],'assistant_prefix':item['prefix'],'input_ids_sha256':hashlib.sha256(json.dumps(item['ids']).encode()).hexdigest(),'semantic_bits':bit_scores,'seconds_batch':seconds}
                    records.append(record)
                    with (RUN/(name+'.jsonl')).open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(json.dumps({'model':name,'mode':mode,'cumulative_records':len(records)}),flush=True)
        reports.append({'model':name,'records':len(records),'load_seconds':load_seconds,'checkpoint_hashes':checkpoint_hashes,'outputs_sha256':hashlib.sha256((RUN/(name+'.jsonl')).read_bytes()).hexdigest()});del model;gc.collect()
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
    (RUN/'PROBE_SEALED.json').write_text(json.dumps({'models':reports,'identity_controls':controls,'source_unchanged':True},indent=2));print('PROBE_SEALED',flush=True)

if __name__=='__main__':run()
