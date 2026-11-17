"""Bounded repair pilot: independent candidate queries, with/without paired margin."""
import probe_frozen as common
from probe_frozen import *
from experiments.relation_solution_data import intervene,targets
from experiments.receipt_dataset import make_world

def query(tokenizer,row,action,pool,nonce):
    text,_=prompt(row['observation'],pool,nonce,question='task',candidate=action)
    text=text.replace('A yes; B no.','Reply ANSWER: followed by 1 if possible, otherwise 0.')
    ids=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
    return text,ids

def data(seed,depth,n):
    w=make_world(seed,('meeting','query')[n%2],depth=depth,noise=2*(n%2),mode='one',allow=False,required=True,delivered=True);w['base_group_id']=w['world_id']
    pair=[targets(w)[0],targets(intervene(w,'swap'))[0]]
    assert pair[0]['optimal_indices']!=pair[1]['optimal_indices']
    return pair

def run():
    out=RUN/'query_pilot';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();source=OLD/'language_models/trace_17/adapter';prior=json.loads((OLD/'language_models/trace_17/sealed.json').read_text())
    train=[data(210100000+n,1,n) for n in range(32)];dev={str(d):[data(210200000+d*100+n,d,n) for n in range(8)] for d in (1,2,3)}
    rng=random.Random(21717);batches=[{'group':rng.randrange(32),'candidate':rng.randrange(2),'nonce':rng.randrange(10**9)} for _ in range(600)]
    plan={'arms':['independent','paired_margin'],'seed':17,'steps':600,'initialization':'unchanged trace_17 adapter, both arms identical','initial_hashes':prior['adapter_hashes'],'train_groups':32,'train_depth':1,'dev_groups_per_depth':8,'dev_depths':[1,2,3],'lr':.0008,'bit_weight':1,'format_and_eos_weight':.1,'paired_margin_weight':1,'paired_margin':1,'batch_pairs':1,'batches':batches,'scope':'DEV relation-producer pilot only; no three-seed or end-to-end-policy claim','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2));(out/'dataset.json').write_text(json.dumps({'train':train,'dev':dev},ensure_ascii=False),encoding='utf-8');reports=[]
    for arm in plan['arms']:
        tokenizer,model=setup(17);assert all(hashlib.sha256((source/p).read_bytes()).hexdigest()==h for p,h in prior['adapter_hashes'].items());set_peft_model_state_dict(model,load_file(str(source/'adapter_model.safetensors')));model.train();pool=palette(tokenizer);binary=[tokenizer.encode(str(i),add_special_tokens=False)[0] for i in (0,1)]
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=0.);folder=out/arm;folder.mkdir();logs=[]
        for step,batch in enumerate(batches):
            def timeout():
                (out/'TIMEOUT.json').write_text(json.dumps({'arm':arm,'step':step,'seconds':9.5}));os._exit(124)
            timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter();pair=train[batch['group']];sequences=[];suffixes=[];labels=[]
            for row in pair:
                c=row['candidates'][batch['candidate']];_,prefix=query(tokenizer,row,c['action'],pool['train'],batch['nonce']);label=c['safe_completion'];suffix=tokenizer.encode('ANSWER:\n'+str(label)+tokenizer.eos_token,add_special_tokens=False)
                assert suffix[-2]==binary[label];sequences.append(prefix+suffix);suffixes.append(suffix);labels.append(label)
            assert labels[0]!=labels[1] and len(sequences[0])==len(sequences[1]) and len(suffixes[0])==len(suffixes[1])
            ids=torch.tensor(sequences);truth=torch.tensor(suffixes);logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),logits_to_keep=len(suffixes[0])+1,use_cache=False).logits[:,:-1]
            ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),truth.reshape(-1),reduction='none').reshape(truth.shape)
            ordinary=ce[:,-2].mean()+.1*torch.cat([ce[:,:-2],ce[:,-1:]],dim=1).mean()
            margins=logits[:,-2,binary[1]]-logits[:,-2,binary[0]];contrast=torch.relu(1-(margins[0]-margins[1])*(2*labels[0]-1))
            loss=ordinary+(contrast if arm=='paired_margin' else 0);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();timer.cancel()
            record={'step':step+1,'loss':float(loss.detach()),'ordinary':float(ordinary.detach()),'contrast':float(contrast.detach()),'seconds':time.perf_counter()-start,'tokens':len(sequences[0])};logs.append(record)
            with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if (step+1)%50==0:print(json.dumps({'arm':arm,**record}),flush=True)
        model.save_pretrained(folder/'adapter');model.eval();outputs=[]
        for depth,pairs in dev.items():
            for n,pair in enumerate(pairs):
                for candidate in range(2):
                    texts=[query(tokenizer,row,row['candidates'][candidate]['action'],pool['fresh'],seed_for(f'query_dev_{depth}_{n}',7331))[0] for row in pair]
                    from lm_common import encode
                    inputs=encode(tokenizer,texts);start=time.perf_counter()
                    with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,max_new_tokens=12,max_time=8.,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                    seconds=time.perf_counter()-start;raws=[tokenizer.decode(x[inputs['input_ids'].shape[1]:],skip_special_tokens=True) for x in generated]
                    with (folder/'dev_raw.jsonl').open('a',encoding='utf-8') as f:
                        for side,raw in enumerate(raws):f.write(json.dumps({'depth':int(depth),'group':n,'candidate':candidate,'side':side,'prompt':texts[side],'raw':raw,'seconds_batch':seconds},ensure_ascii=False)+'\n')
                    for side,(row,raw) in enumerate(zip(pair,raws)):
                        matches=re.findall(r'ANSWER:\s*([01])\b',raw);valid=len(matches)==1 and seconds<9.5;selected=matches[0] if valid else None
                        outputs.append({'pair_id':f'd{depth}_g{n}_c{candidate}','side':side,'depth':int(depth),'selected':selected,'optimal_action':str(row['candidates'][candidate]['safe_completion']),'actions':['0','1']})
        (folder/'dev_scored.json').write_text(json.dumps(outputs,indent=2));from kci.action_binding_metrics import pair_metrics
        report={'arm':arm,'steps':600,'max_update_seconds':max(r['seconds'] for r in logs),'depths':{str(d):pair_metrics([r for r in outputs if r['depth']==d]) for d in (1,2,3)},'adapter_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'adapter').iterdir() if p.is_file()}}
        reports.append(report);(folder/'SEALED.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);del model,opt;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'models':reports,'pilot_only':True,'whole_goal_complete':False},indent=2))

if __name__=='__main__':run()
