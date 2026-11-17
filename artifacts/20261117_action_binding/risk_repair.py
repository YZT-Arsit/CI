"""Matched small safety-utility repair: add immediate-safety probability loss."""
from coverage_repair import *

def risk_loss(tokenizer,model,pair,pool,nonce,step,weight):
    texts=[];answers=[];indices=[];labels=[];letters=answer_ids(tokenizer);binary={tokenizer.encode(str(i),add_special_tokens=False)[0]:str(i) for i in (0,1)}
    for side,row in enumerate(pair):
        for candidate in row['candidates']:
            texts.append(query(tokenizer,row,candidate['action'],pool,nonce)[0]);answers.append('ANSWER:\n'+str(candidate['safe_completion']));labels.append(candidate['safe_completion']);indices.append((side,candidate['action']))
    qlogits,qce,qtokens=forward_targets(tokenizer,model,texts,answers);estimates=[{},{}]
    for (side,action),value in zip(indices,qlogits[:,-2].detach().argmax(-1).tolist()):estimates[side][action]=binary.get(value,'unknown')
    positive=torch.tensor(labels,dtype=torch.bool);bit=.5*(qce[positive,-2].mean()+qce[~positive,-2].mean()) if positive.any() and (~positive).any() else qce[:,-2].mean();qloss=bit+.1*torch.cat([qce[:,:-2],qce[:,-1:]],1).mean();texts=[];answers=[];goodsets=[];safesets=[];orders=[]
    for side,row in enumerate(pair):
        text,actions=actor_text(row['observation'],pool,nonce,estimates[side]);answer,good=response(row,actions,pool,nonce,'plain',random.Random(step*2+side));texts.append(text);answers.append(answer);goodsets.append(good);orders.append(actions);safe={c['action'] for c in row['candidates'] if c['immediate_safe']};safesets.append([i for i,a in enumerate(actions) if a in safe]);assert safesets[-1]
    logits,ce,atokens=forward_targets(tokenizer,model,texts,answers);choice=torch.stack([torch.logsumexp(logits[i,-2],0)-torch.logsumexp(logits[i,-2,letters[good]],0) for i,good in enumerate(goodsets)]).mean();aloss=choice+.1*torch.cat([ce[:,:-2],ce[:,-1:]],1).mean()
    safety=torch.stack([torch.logsumexp(logits[i,-2,letters[:len(order)]],0)-torch.logsumexp(logits[i,-2,letters[safe]],0) for i,(order,safe) in enumerate(zip(orders,safesets))]).mean()
    return qloss,aloss,safety,qloss+aloss+weight*safety,qtokens+atokens

def dev():
    worlds={}
    for condition in ('base','base_swap','depth2','depth2_swap','depth3','depth3_swap','optional','authorized','draft','both'):
        source=condition.removesuffix('_swap');offset={'base':0,'depth2':1000,'depth3':2000,'optional':3000,'authorized':4000,'draft':5000,'both':6000}[source];rows=[]
        for n in range(8):
            w=make_world(240000000+offset+n,('meeting','query')[n%2],depth={'depth2':2,'depth3':3}.get(source,1),noise=2,mode='both' if source=='both' else 'one',allow=source=='authorized',required=source!='optional',delivered=source!='draft');w['base_group_id']=w['world_id']
            if condition.endswith('_swap'):w=intervene(w,'swap')
            rows.append(w)
        worlds[condition]=rows
    return worlds

def run():
    out=RUN/'risk_repair';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();buckets=full_buckets();rng=random.Random(21817);batches=[]
    for step in range(100):
        role=('Planner','Source','Worker','Worker')[step%4];keys=sorted(k for k in buckets if k[0]==role);key=rng.choice(keys);batches.append({'key':list(key),'index':rng.randrange(len(buckets[key])),'nonce':rng.randrange(10**9)})
    folder=RUN/'coverage_repair/class_balanced';seal=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());worlds=dev();(out/'dev_worlds.json').write_text(json.dumps(worlds,ensure_ascii=False),encoding='utf-8');plan={'arms':['utility','safety_utility'],'evidence_arms':['utility','safety_utility'],'steps':100,'seed':17,'lr':.0008,'batches':batches,'immediate_safety_loss_weights':{'utility':0.,'safety_utility':1.},'loss_definition':'minus log safe mass conditional on legal action letters; training loss only, no inference mask','initial_adapter_hashes':seal['adapter_hashes'],'query_class_balance':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'new DEV pilot motivated by unsafe class-balanced outputs; not a final or three-seed safety guarantee','oracle_at_runtime':False};(out/'PLAN.json').write_text(json.dumps(plan,indent=2));summaries=[]
    while not (RUN/'composition_repair_v2/RESULTS.json').exists():time.sleep(2)
    for arm in plan['arms']:
        tokenizer,model=setup(17);assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));pool=palette(tokenizer);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0008,weight_decay=0.);target=out/arm;target.mkdir();tokens_total=0
        for step,batch in enumerate(batches):
            def timeout():
                (target/'TIMEOUT.json').write_text(json.dumps({'step':step,'seconds':9.5}));os._exit(124)
            timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter();pair=buckets[tuple(batch['key'])][batch['index']];q,a,s,loss,tokens=risk_loss(tokenizer,model,pair,pool['train'],batch['nonce'],step,plan['immediate_safety_loss_weights'][arm]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();timer.cancel();seconds=time.perf_counter()-start;tokens_total+=tokens;record={'step':step+1,'query_loss':float(q.detach()),'actor_loss':float(a.detach()),'safety_loss':float(s.detach()),'seconds':seconds,'input_tokens':tokens}
            with (target/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if (step+1)%25==0:print(json.dumps({'arm':arm,**record}),flush=True)
        model.save_pretrained(target/'adapter');(target/'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target/'adapter').iterdir() if p.is_file()},'steps':100,'input_tokens':tokens_total},indent=2));model.eval()
        for condition,ws in worlds.items():
            rows=evaluate(tokenizer,model,ws,pool['fresh'],True);path=target/(condition+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');summary={'arm':arm,'condition':condition,'episodes':len(rows),**{k:sum(r[k] for r in rows) for k in ('safe_success','strict_success','invalid','invalid_queries')},'valid_ci':sum(r['ci'] and not r['invalid'] for r in rows),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};summaries.append(summary);print(json.dumps(summary),flush=True)
        del model,opt;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))

if __name__=='__main__':run()
