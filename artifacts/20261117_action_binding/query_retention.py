"""Post-hoc query retention diagnostic after actor-only training; no new tuning."""
from probe_frozen import *
from query_pilot import query
from lm_common import encode
from kci.action_binding_metrics import pair_metrics

def run():
    out=RUN/'query_retention';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();data=json.loads((RUN/'query_pilot/dataset.json').read_text());(out/'PLAN.json').write_text(json.dumps({'models':['direct','query_warmup'],'scope':'post-hoc reuse of query DEV for retention diagnosis only','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256((RUN/'query_pilot/dataset.json').read_bytes()).hexdigest()},indent=2))
    while not (RUN/'LEARNED_EVIDENCE_SMOKE.json').exists():time.sleep(2)
    summaries=[]
    for arm in ('direct','query_warmup'):
        folder=RUN/'actor_transfer'/arm;seal=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['fresh'];outputs=[]
        for depth,pairs in data['dev'].items():
            scored=[]
            for n,pair in enumerate(pairs):
                for candidate in (0,1):
                    texts=[query(tokenizer,row,row['candidates'][candidate]['action'],pool,seed_for(f'query_dev_{depth}_{n}',7331))[0] for row in pair];inputs=encode(tokenizer,texts);start=time.perf_counter()
                    with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,max_new_tokens=12,max_time=8.,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                    seconds=time.perf_counter()-start
                    for side,row in enumerate(pair):
                        raw=tokenizer.decode(generated[side,inputs['input_ids'].shape[1]:],skip_special_tokens=True);matches=re.findall(r'ANSWER:\s*([01])\b',raw);selected=matches[0] if len(matches)==1 and seconds<9.5 else None
                        record={'pair_id':f'd{depth}_g{n}_c{candidate}','side':side,'depth':int(depth),'selected':selected,'optimal_action':str(row['candidates'][candidate]['safe_completion']),'actions':['0','1'],'prompt':texts[side],'raw':raw,'seconds_batch':seconds};outputs.append(record);scored.append(record)
            summary={'arm':arm,'depth':int(depth),'pairs':pair_metrics(scored)};summaries.append(summary);print(json.dumps(summary),flush=True)
        path=out/(arm+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs),encoding='utf-8');(out/(arm+'_SEALED.json')).write_text(json.dumps({'adapter_hashes':seal['adapter_hashes'],'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2));del model;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))

if __name__=='__main__':run()
