"""Equal-update DEV test: direct action learning versus relation warm-up transfer."""
import probe_frozen as common
from probe_frozen import *
from lm_common import load_states
from evaluate_language import evaluate
from experiments.relation_solution_data import intervene
from experiments.receipt_dataset import make_world

def paired_states():
    groups={}
    for row in load_states('train'):
        obs=row['observation']
        if len(obs['contract']['providers'][0]['links'])!=1:continue
        key=(row['base_group_id'],obs['role'],obs['committed_ticket']);groups.setdefault(key,[]).append(row)
    buckets={}
    for pair in groups.values():
        assert len(pair)==2
        if all(len(r['optimal_indices'])==len(r['candidates']) for r in pair):continue
        obs=pair[0]['observation'];contract=obs['contract']
        key=(obs['role'],contract['permission'],contract['binding_required'],any(c['safe_completion'] for r in pair for c in r['candidates']),any(not c['immediate_safe'] for r in pair for c in r['candidates']))
        buckets.setdefault(key,[]).append(sorted(pair,key=lambda r:r['world_id']))
    return buckets

def dev_worlds():
    conditions=['base','base_swap','depth2','depth2_swap','depth3','depth3_swap','optional','authorized','draft','both','broken','reverse'];result={}
    for condition in conditions:
        source=condition.removesuffix('_swap');index=conditions.index(source);rows=[]
        for n in range(16):
            kw={'depth':{'depth2':2,'depth3':3}.get(source,1),'noise':2,'allow':source=='authorized','required':source!='optional','delivered':source!='draft','mode':'both' if source=='both' else 'one'}
            w=make_world(212000000+index*1000+n,('meeting','query')[n%2],**kw);w['base_group_id']=w['world_id']
            if source in ('broken','reverse'):w=intervene(w,source)
            if condition.endswith('_swap'):w=intervene(w,'swap')
            rows.append(w)
        result[condition]=rows
    return result

def run():
    out=RUN/'actor_transfer';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists()
    while not (RUN/'query_pilot/RESULTS.json').exists():time.sleep(2)
    buckets=paired_states();keys=sorted(buckets);rng=random.Random(21737);batches=[]
    for step in range(1200):
        key=rng.choice(keys);batches.append({'key':list(key),'index':rng.randrange(len(buckets[key])),'nonce':rng.randrange(10**9)})
    conditions=dev_worlds();(out/'dev_worlds.json').write_text(json.dumps(conditions,ensure_ascii=False),encoding='utf-8')
    plan={'seed':17,'arms':{'direct':{'initialization':'old trace_17','actor_steps':1200,'query_steps':0},'query_warmup':{'initialization':'query_pilot independent','actor_steps':600,'query_steps':600}},'total_added_updates_per_arm':1200,'same_last_600_actor_batches':True,'batches':batches,'lr':.0008,'decision_weight':1.,'format_and_eos_weight':.1,'train_depth':1,'train_roles':['Planner','Source','Worker'],'candidate_ties':'sum probability over all optimal actions','buckets':[{ 'key':list(k),'pairs':len(v)} for k,v in buckets.items()],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'seed17 DEV experiment only; no long-chain or final-policy promotion','no_teacher_prefix_at_evaluation':True,'equal_updates_not_equal_token_budget':True}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2));summaries=[]
    for arm in ('direct','query_warmup'):
        tokenizer,model=setup(17);pool=palette(tokenizer);letters=answer_ids(tokenizer)
        if arm=='direct':folder=OLD/'language_models/trace_17';seal=json.loads((folder/'sealed.json').read_text());start_step=0
        else:folder=RUN/'query_pilot/independent';seal=json.loads((folder/'SEALED.json').read_text());start_step=600
        assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')))
        target=out/arm;target.mkdir();(target/'INITIAL.json').write_text(json.dumps({'adapter_hashes':seal['adapter_hashes'],'start_step':start_step},indent=2));model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0008,weight_decay=0.);token_total=0;max_seconds=0
        for step in range(start_step,1200):
            def timeout():
                (out/'TIMEOUT.json').write_text(json.dumps({'arm':arm,'step':step,'seconds':9.5}));os._exit(124)
            timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter();batch=batches[step];pair=buckets[tuple(batch['key'])][batch['index']];seq=[];suffixes=[];goodsets=[]
            for side,row in enumerate(pair):
                text,actions=prompt(row['observation'],pool['train'],batch['nonce']);text+='\nReply ANSWER:\nletter.';prefix=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
                answer,good=response(row,actions,pool['train'],batch['nonce'],'plain',random.Random(step*2+side));suffix=tokenizer.encode(answer+tokenizer.eos_token,add_special_tokens=False);assert suffix[-2] in letters.tolist();seq.append(prefix+suffix);suffixes.append(suffix);goodsets.append(good);token_total+=len(suffix)
            length=max(map(len,seq));keep=max(map(len,suffixes))+1;ids=torch.tensor([[tokenizer.pad_token_id]*(length-len(x))+x for x in seq]);mask=torch.tensor([[0]*(length-len(x))+[1]*len(x) for x in seq]);truth=torch.tensor(suffixes)
            logits=model(input_ids=ids,attention_mask=mask,logits_to_keep=keep,use_cache=False).logits[:,:-1];ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),truth.reshape(-1),reduction='none').reshape(truth.shape)
            choice=torch.stack([torch.logsumexp(logits[i,-2],0)-torch.logsumexp(logits[i,-2,letters[good]],0) for i,good in enumerate(goodsets)]).mean();loss=choice+.1*torch.cat([ce[:,:-2],ce[:,-1:]],1).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();timer.cancel();seconds=time.perf_counter()-start;max_seconds=max(max_seconds,seconds)
            row={'global_step':step+1,'actor_step':step-start_step+1,'loss':float(loss.detach()),'seconds':seconds,'target_tokens':sum(map(len,suffixes))}
            with (target/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if (step+1)%100==0:print(json.dumps({'arm':arm,**row}),flush=True)
        model.save_pretrained(target/'adapter');checkpoint={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target/'adapter').iterdir() if p.is_file()};(target/'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes':checkpoint,'actor_steps':1200-start_step,'actor_target_tokens':token_total,'max_update_seconds':max_seconds},indent=2));model.eval()
        for condition,worlds in conditions.items():
            rows=evaluate(tokenizer,model,'plain',worlds,pool['fresh']);path=target/(condition+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');summary={'arm':arm,'condition':condition,'episodes':len(rows),'safe_success':sum(r['safe_success'] for r in rows),'valid_ci':sum(r['ci'] and not r['invalid'] for r in rows),'invalid':sum(r['invalid'] for r in rows),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};summaries.append(summary);print(json.dumps(summary),flush=True)
        del model,opt;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))

if __name__=='__main__':run()
