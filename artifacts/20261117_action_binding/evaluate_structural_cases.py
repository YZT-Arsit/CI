"""Frozen internal-edge DEV challenge; query readout and full actor execution."""
from probe_frozen import *
from query_pilot import query
from lm_common import encode
from evaluate_language import evaluate
from kci.action_binding_metrics import pair_metrics

def run():
    out=RUN/'structural_cases';assert not (out/'EVAL_PLAN.json').exists();plan=json.loads((out/'PLAN.json').read_text());assert hashlib.sha256((out/'dataset.json').read_bytes()).hexdigest()==plan['dataset_sha256']
    names=['query_independent','query_paired_margin','actor_direct','actor_query_warmup'];(out/'EVAL_PLAN.json').write_text(json.dumps({'models':names,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dataset_sha256':plan['dataset_sha256'],'scope':'additional DEV challenge; no training or model selection','no_teacher_or_mask':True},indent=2))
    while not (RUN/'actor_transfer/RESULTS.json').exists():time.sleep(2)
    data=json.loads((out/'dataset.json').read_text(encoding='utf-8'));summaries=[]
    for name in names:
        query_arm=name.startswith('query_');folder=RUN/('query_pilot' if query_arm else 'actor_transfer')/name.split('_',1)[1];seal=json.loads((folder/('SEALED.json' if query_arm else 'CHECKPOINT_SEALED.json')).read_text());assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items())
        tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['fresh'];outputs=[]
        for depth in (3,4,6):
            ws=[w for w in data['worlds'] if len(w['providers'][0]['links'])==depth]
            if not query_arm:
                rows=evaluate(tokenizer,model,'plain',ws,pool);outputs.extend(rows)
                summaries.append({'model':name,'depth':depth,'world_pairs':len(ws)//2,'both_safe_success':sum(a['safe_success'] and b['safe_success'] for a,b in zip(rows[::2],rows[1::2])),'safe_success':sum(r['safe_success'] for r in rows),'invalid':sum(r['invalid'] for r in rows)})
            else:
                states=[r for r in data['states'] if len(r['observation']['contract']['providers'][0]['links'])==depth];scored=[]
                for n in range(0,len(states),2):
                    pair=states[n:n+2];nonce=seed_for(pair[0]['base_group_id'],7331)
                    for candidate in (0,1):
                        texts=[query(tokenizer,r,r['candidates'][candidate]['action'],pool,nonce)[0] for r in pair];inputs=encode(tokenizer,texts);start=time.perf_counter()
                        with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,max_new_tokens=12,max_time=8.,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                        seconds=time.perf_counter()-start
                        for side,row in enumerate(pair):
                            raw=tokenizer.decode(generated[side,inputs['input_ids'].shape[1]:],skip_special_tokens=True);matches=re.findall(r'ANSWER:\s*([01])\b',raw);selected=matches[0] if len(matches)==1 and seconds<9.5 else None
                            record={'pair_id':f"{row['base_group_id']}_c{candidate}",'side':side,'depth':depth,'selected':selected,'optimal_action':str(row['candidates'][candidate]['safe_completion']),'actions':['0','1'],'prompt':texts[side],'raw':raw,'seconds_batch':seconds};outputs.append(record);scored.append(record)
                summaries.append({'model':name,'depth':depth,'candidate_pairs':pair_metrics(scored)})
            print(json.dumps(summaries[-1]),flush=True)
        path=out/(name+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs),encoding='utf-8');(out/(name+'_SEALED.json')).write_text(json.dumps({'adapter_hashes':seal['adapter_hashes'],'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2));del model;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))

if __name__=='__main__':run()
