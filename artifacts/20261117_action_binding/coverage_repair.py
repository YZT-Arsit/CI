"""Retain tied-state query supervision; isolate query class balance on matched batches."""
from learned_evidence import *
from lm_common import load_states

def full_buckets():
    groups={};buckets={}
    for row in load_states('train'):
        obs=row['observation']
        if len(obs['contract']['providers'][0]['links'])!=1:continue
        groups.setdefault((row['base_group_id'],obs['role'],obs['committed_ticket']),[]).append(row)
    for pair in groups.values():
        assert len(pair)==2;obs=pair[0]['observation'];c=obs['contract'];key=(obs['role'],c['permission'],c['binding_required'],any(x['safe_completion'] for r in pair for x in r['candidates']),any(not x['immediate_safe'] for r in pair for x in r['candidates']));buckets.setdefault(key,[]).append(sorted(pair,key=lambda r:r['world_id']))
    return buckets

def loss_for(tokenizer,model,pair,pool,nonce,step,balanced):
    texts=[];answers=[];indices=[];labels=[];letters=answer_ids(tokenizer);binary={tokenizer.encode(str(i),add_special_tokens=False)[0]:str(i) for i in (0,1)}
    for side,row in enumerate(pair):
        for candidate in row['candidates']:
            texts.append(query(tokenizer,row,candidate['action'],pool,nonce)[0]);answers.append('ANSWER:\n'+str(candidate['safe_completion']));labels.append(candidate['safe_completion']);indices.append((side,candidate['action']))
    qlogits,qce,qtokens=forward_targets(tokenizer,model,texts,answers);estimates=[{},{}]
    for (side,action),value in zip(indices,qlogits[:,-2].detach().argmax(-1).tolist()):estimates[side][action]=binary.get(value,'unknown')
    positive=torch.tensor(labels,dtype=torch.bool);bit_loss=.5*(qce[positive,-2].mean()+qce[~positive,-2].mean()) if balanced and positive.any() and (~positive).any() else qce[:,-2].mean();qloss=bit_loss+.1*torch.cat([qce[:,:-2],qce[:,-1:]],1).mean();texts=[];answers=[];goodsets=[]
    for side,row in enumerate(pair):
        text,actions=actor_text(row['observation'],pool,nonce,estimates[side]);answer,good=response(row,actions,pool,nonce,'plain',random.Random(step*2+side));texts.append(text);answers.append(answer);goodsets.append(good)
    logits,ce,atokens=forward_targets(tokenizer,model,texts,answers);aloss=torch.stack([torch.logsumexp(logits[i,-2],0)-torch.logsumexp(logits[i,-2,letters[good]],0) for i,good in enumerate(goodsets)]).mean()+.1*torch.cat([ce[:,:-2],ce[:,-1:]],1).mean()
    return qloss,aloss,qtokens+atokens,estimates,{'positive':sum(labels),'negative':len(labels)-sum(labels)}

def new_dev():
    result={}
    for condition in ('base','base_swap','depth2','depth2_swap','depth3','depth3_swap','optional','authorized','draft','both'):
        source=condition.removesuffix('_swap');offset={'base':0,'depth2':1000,'depth3':2000,'optional':3000,'authorized':4000,'draft':5000,'both':6000}[source];rows=[]
        for n in range(8):
            w=make_world(219000000+offset+n,('meeting','query')[n%2],depth={'depth2':2,'depth3':3}.get(source,1),noise=2,mode='both' if source=='both' else 'one',allow=source=='authorized',required=source!='optional',delivered=source!='draft');w['base_group_id']=w['world_id']
            if condition.endswith('_swap'):w=intervene(w,'swap')
            rows.append(w)
        result[condition]=rows
    return result

def run():
    out=RUN/'coverage_repair';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();buckets=full_buckets();rng=random.Random(21777);batches=[]
    for step in range(200):
        role=('Planner','Source','Worker','Worker')[step%4];keys=sorted(k for k in buckets if k[0]==role);key=rng.choice(keys);batches.append({'key':list(key),'index':rng.randrange(len(buckets[key])),'nonce':rng.randrange(10**9)})
    folder=RUN/'learned_evidence/evidence';seal=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());worlds=new_dev();(out/'dev_worlds.json').write_text(json.dumps(worlds,ensure_ascii=False),encoding='utf-8');plan={'arms':['coverage_only','class_balanced'],'evidence_arms':['coverage_only','class_balanced'],'steps':200,'seed':17,'lr':.0008,'batches':batches,'retain_all_tied_states_for_queries':True,'actor_ties':'sum probability over all optimal options; no fabricated strict preference','role_updates':{'Planner':50,'Source':50,'Worker':100},'buckets':[{'key':list(k),'pairs':len(v)} for k,v in sorted(buckets.items())],'initial_adapter_hashes':seal['adapter_hashes'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'new DEV pilot; two arms differ only in positive/negative query loss averaging, both have repaired coverage','no_effect_of_coverage_alone_claim':'previous run also differs in update budget and role schedule','oracle_at_runtime':False};(out/'PLAN.json').write_text(json.dumps(plan,indent=2));summaries=[]
    for arm in plan['arms']:
        tokenizer,model=setup(17);assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));pool=palette(tokenizer);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0008,weight_decay=0.);target=out/arm;target.mkdir();tokens_total=0
        for step,batch in enumerate(batches):
            def timeout():
                (target/'TIMEOUT.json').write_text(json.dumps({'step':step,'seconds':9.5}));os._exit(124)
            timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter();pair=buckets[tuple(batch['key'])][batch['index']];q,a,tokens,estimates,counts=loss_for(tokenizer,model,pair,pool['train'],batch['nonce'],step,arm=='class_balanced');opt.zero_grad();(q+a).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();timer.cancel();seconds=time.perf_counter()-start;tokens_total+=tokens;record={'step':step+1,'query_loss':float(q.detach()),'actor_loss':float(a.detach()),'seconds':seconds,'input_tokens':tokens,'query_labels':counts,'predicted_estimates':estimates}
            with (target/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if (step+1)%50==0:print(json.dumps({'arm':arm,**{k:v for k,v in record.items() if k!='predicted_estimates'}}),flush=True)
        model.save_pretrained(target/'adapter');(target/'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target/'adapter').iterdir() if p.is_file()},'steps':200,'input_tokens':tokens_total},indent=2));model.eval()
        for condition,ws in worlds.items():
            rows=evaluate(tokenizer,model,ws,pool['fresh'],True);path=target/(condition+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');summary={'arm':arm,'condition':condition,'episodes':len(rows),**{k:sum(r[k] for r in rows) for k in ('safe_success','strict_success','invalid','invalid_queries')},'valid_ci':sum(r['ci'] and not r['invalid'] for r in rows),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};summaries.append(summary);print(json.dumps(summary),flush=True)
        del model,opt;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))

if __name__=='__main__':run()
