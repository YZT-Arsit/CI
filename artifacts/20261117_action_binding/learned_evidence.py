"""Shared-weight query-to-action pilot; predicted estimates only in actor inputs."""
from probe_frozen import *
from actor_transfer import paired_states
from query_pilot import query
from lm_common import encode
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import intervene,environment
from dataclasses import asdict

def actor_text(obs,pool,nonce,estimates=None):
    text,actions=prompt(obs,pool,nonce)
    if estimates is not None:
        text+='Own estimates (1 means compliant completion remains possible): '+ '; '.join(chr(65+i)+'='+estimates[a] for i,a in enumerate(actions))+'.\n'
    return text+'\nReply ANSWER:\nletter.',actions

def forward_targets(tokenizer,model,texts,answers):
    seq=[];suffix=[]
    for text,answer in zip(texts,answers):
        start=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True);end=tokenizer.encode(answer+tokenizer.eos_token,add_special_tokens=False);seq.append(start+end);suffix.append(end)
    assert len({len(x) for x in suffix})==1
    width=max(map(len,seq));ids=torch.tensor([[tokenizer.pad_token_id]*(width-len(x))+x for x in seq]);mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in seq]);truth=torch.tensor(suffix)
    logits=model(input_ids=ids,attention_mask=mask,logits_to_keep=len(suffix[0])+1,use_cache=False).logits[:,:-1]
    ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),truth.reshape(-1),reduction='none').reshape(truth.shape)
    return logits,ce,sum(map(len,seq))

def generate(tokenizer,model,texts):
    inputs=encode(tokenizer,texts);start=time.perf_counter()
    with torch.inference_mode():out=model.generate(**inputs,do_sample=False,max_new_tokens=16,max_time=8.,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    elapsed=time.perf_counter()-start
    return [tokenizer.decode(x[inputs['input_ids'].shape[1]:],skip_special_tokens=True) for x in out],elapsed

def evaluate(tokenizer,model,worlds,pool,evidence):
    envs=[environment(w) for w in worlds];states=[e.initial() for e in envs];traces=[[] for w in worlds];failed=[False]*len(worlds);bad_query=[0]*len(worlds)
    for phase in range(3):
        for start in range(0,len(worlds),2):
            active=[i for i in range(start,min(start+2,len(worlds))) if not states[i].done and not failed[i]]
            if not active:continue
            observations={i:envs[i].observation(states[i]) for i in active};nonces={i:seed_for(worlds[i]['base_group_id'],phase) for i in active};estimates={i:{} for i in active};query_records={i:[] for i in active}
            if evidence:
                queries=[];index=[]
                for i in active:
                    for action in observations[i]['actions']:
                        queries.append(query(tokenizer,{'observation':observations[i]},action,pool,nonces[i])[0]);index.append((i,action))
                raws,seconds=generate(tokenizer,model,queries)
                for (i,action),text,raw in zip(index,queries,raws):
                    matches=re.findall(r'ANSWER:\s*([01])\b',raw);valid=len(matches)==1 and seconds<9.5;bit=matches[0] if valid else 'unknown';estimates[i][action]=bit;bad_query[i]+=int(not valid)
                    query_records[i].append({'action':action,'prompt':text,'raw':raw,'estimate':bit,'valid':valid,'seconds_batch':seconds})
            texts=[];orders=[]
            for i in active:
                text,actions=actor_text(observations[i],pool,nonces[i],estimates[i] if evidence else None);texts.append(text);orders.append(actions)
            raws,seconds=generate(tokenizer,model,texts)
            for i,text,actions,raw in zip(active,texts,orders,raws):
                matches=re.findall(r'ANSWER:\s*([ABC])\b',raw);valid=len(matches)==1 and ord(matches[0])-65<len(actions) and seconds<9.5
                record={'phase':phase,'role':envs[i].actor(states[i]),'prompt':text,'actions':actions,'raw':raw,'seconds_batch':seconds,'format_valid':valid,'selected':None,'queries':query_records[i]}
                if valid:
                    action=actions[ord(matches[0])-65];states[i],audit=envs[i].transition(states[i],action);record.update(selected=action,audit=audit);failed[i]=audit['verdict']=='UNKNOWN'
                else:failed[i]=True
                traces[i].append(record)
    return [{'world_id':w['world_id'],'base_group_id':w['base_group_id'],'ci':not s.violation,'safe_success':s.success and not s.violation and not failed[i],'strict_success':s.success and not s.violation and not failed[i] and not bad_query[i],'invalid':failed[i],'invalid_queries':bad_query[i],'trace':traces[i],'final':asdict(s),'policy_interventions':0} for i,(w,s) in enumerate(zip(worlds,states))]

def dev_worlds():
    result={}
    for condition in ('base','base_swap','depth2','depth2_swap','depth3','depth3_swap','optional','authorized','draft','both'):
        source=condition.removesuffix('_swap');offset={'base':0,'depth2':1000,'depth3':2000,'optional':3000,'authorized':4000,'draft':5000,'both':6000}[source];rows=[]
        for n in range(8):
            w=make_world(214000000+offset+n,('meeting','query')[n%2],depth={'depth2':2,'depth3':3}.get(source,1),noise=2,mode='both' if source=='both' else 'one',allow=source=='authorized',required=source!='optional',delivered=source!='draft');w['base_group_id']=w['world_id']
            if condition.endswith('_swap'):w=intervene(w,'swap')
            rows.append(w)
        result[condition]=rows
    return result

def joint_loss(tokenizer,model,pair,pool,nonce,step,evidence):
    texts=[];answers=[];indices=[];letters=answer_ids(tokenizer);binary={tokenizer.encode(str(i),add_special_tokens=False)[0]:str(i) for i in (0,1)}
    for side,row in enumerate(pair):
        for candidate in row['candidates']:
            texts.append(query(tokenizer,row,candidate['action'],pool,nonce)[0]);answers.append('ANSWER:\n'+str(candidate['safe_completion']));indices.append((side,candidate['action']))
    qlogits,qce,qtokens=forward_targets(tokenizer,model,texts,answers);estimates=[{},{}]
    for (side,action),value in zip(indices,qlogits[:,-2].detach().argmax(-1).tolist()):estimates[side][action]=binary.get(value,'unknown')
    qloss=qce[:,-2].mean()+.1*torch.cat([qce[:,:-2],qce[:,-1:]],1).mean();texts=[];answers=[];goodsets=[]
    for side,row in enumerate(pair):
        text,actions=actor_text(row['observation'],pool,nonce,estimates[side] if evidence else None);answer,good=response(row,actions,pool,nonce,'plain',random.Random(step*2+side));texts.append(text);answers.append(answer);goodsets.append(good)
    logits,ce,atokens=forward_targets(tokenizer,model,texts,answers);aloss=torch.stack([torch.logsumexp(logits[i,-2],0)-torch.logsumexp(logits[i,-2,letters[good]],0) for i,good in enumerate(goodsets)]).mean()+.1*torch.cat([ce[:,:-2],ce[:,-1:]],1).mean()
    return qloss,aloss,qtokens+atokens,estimates

def run():
    out=RUN/'learned_evidence';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();buckets=paired_states();rng=random.Random(21757);batches=[]
    for step in range(400):
        role=('Planner','Planner','Source','Worker')[step%4];keys=sorted(k for k in buckets if k[0]==role);key=rng.choice(keys);batches.append({'key':list(key),'index':rng.randrange(len(buckets[key])),'nonce':rng.randrange(10**9)})
    worlds=dev_worlds();(out/'dev_worlds.json').write_text(json.dumps(worlds,ensure_ascii=False),encoding='utf-8');folder=RUN/'query_pilot/independent';seal=json.loads((folder/'SEALED.json').read_text())
    plan={'arms':['auxiliary','evidence'],'seed':17,'steps':400,'lr':.0008,'query_weight':1.,'action_weight':1.,'format_weight':.1,'train_depth':1,'role_updates':{'Planner':200,'Source':100,'Worker':100},'batches':batches,'initial_adapter_hashes':seal['adapter_hashes'],'same_shared_weights_for_queries_and_actor':True,'actor_training_inputs':'predictions from query logits under ANSWER format prefix, no true bits','evaluation':'free query generation and free action generation','oracle_at_runtime':False,'baseline_commit':'b476787ba9e0cebf19d6b81569eeaf36b0570fbc','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'seed17 DEV pilot only; not a complete solution'}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2));summaries=[]
    while not (RUN/'structural_cases/RESULTS.json').exists():time.sleep(2)
    for arm in plan['arms']:
        tokenizer,model=setup(17);assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.train();pool=palette(tokenizer);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0008,weight_decay=0.);target=out/arm;target.mkdir();token_total=0
        for step,batch in enumerate(batches):
            def timeout():
                (target/'TIMEOUT.json').write_text(json.dumps({'step':step,'seconds':9.5}));os._exit(124)
            timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter();pair=buckets[tuple(batch['key'])][batch['index']]
            qloss,aloss,tokens,estimates=joint_loss(tokenizer,model,pair,pool['train'],batch['nonce'],step,arm=='evidence');loss=qloss+aloss
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();timer.cancel();seconds=time.perf_counter()-start;token_total+=tokens
            record={'step':step+1,'query_loss':float(qloss.detach()),'actor_loss':float(aloss.detach()),'seconds':seconds,'input_tokens':tokens,'predicted_estimates':estimates}
            with (target/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if (step+1)%50==0:print(json.dumps({'arm':arm,**{k:v for k,v in record.items() if k!='predicted_estimates'}}),flush=True)
        model.save_pretrained(target/'adapter');(target/'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target/'adapter').iterdir() if p.is_file()},'steps':400,'input_tokens':token_total},indent=2));model.eval()
        for condition,ws in worlds.items():
            rows=evaluate(tokenizer,model,ws,pool['fresh'],arm=='evidence');path=target/(condition+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');summary={'arm':arm,'condition':condition,'episodes':len(rows),**{k:sum(r[k] for r in rows) for k in ('safe_success','strict_success','invalid','invalid_queries')},'valid_ci':sum(r['ci'] and not r['invalid'] for r in rows),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};summaries.append(summary);print(json.dumps(summary),flush=True)
        del model,opt;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))

if __name__=='__main__':run()
