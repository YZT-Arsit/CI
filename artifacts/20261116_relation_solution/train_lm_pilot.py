"""Train/dev-only language pilot; main test files are never loaded."""
from lm_common import *
import time,threading,hashlib

def run(arm,seed,steps):
    out=R/'lm_pilots'/f'{arm}_{seed}_{steps}';out.mkdir(parents=True,exist_ok=False)
    tokenizer,model=setup(seed);names=palette(tokenizer);letters=answer_ids(tokenizer);rows=load_states('train')
    rng=random.Random(seed);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0008,weight_decay=0.)
    groups={}
    for row in rows:
        key=(row['observation']['role'],row['no_safe_completion']);groups.setdefault(key,[]).append(row)
    keys=list(groups);logs=[]
    (out/'plan.json').write_text(json.dumps({'arm':arm,'seed':seed,'steps':steps,'lr':.0008,'batch':2,'data':'train only','reference':'base model, LoRA disabled','selection':'pilot dev only'},indent=2))
    for step in range(steps):
        timer=threading.Timer(9.5,lambda:os._exit(124));timer.daemon=True;timer.start();start=time.perf_counter()
        texts=[];targets=[];rejects=[]
        for b in range(2):
            row=rng.choice(groups[rng.choice(keys)]);obs=row['observation'];nonce=rng.randrange(10**9)
            question=('safe' if step%4==0 else 'value' if step%4==1 else 'policy') if arm=='joint' else 'policy'
            if question=='policy':
                text,actions=prompt(obs,names['train'],nonce)
                good=[actions.index(row['candidates'][i]['action']) for i in row['optimal_indices']]
                bad=[i for i in range(len(actions)) if i not in good];targets.append(rng.choice(good));rejects.append(rng.choice(bad) if bad else -1)
            else:
                c=rng.choice(row['candidates']);text,_=prompt(obs,names['train'],nonce,question,c['action'])
                value=c['immediate_safe'] if question=='safe' else c['safe_completion'];targets.append(0 if value else 1);rejects.append(1-targets[-1])
            texts.append(text)
        inputs=encode(tokenizer,texts);target=torch.tensor(targets);reject=torch.tensor([max(0,x) for x in rejects]);valid=torch.tensor([x>=0 for x in rejects])
        with torch.no_grad(),model.disable_adapter():ref=model_logits(model,inputs)[:,letters]
        logits=model_logits(model,inputs);scores=logits[:,letters]
        sft=torch.nn.functional.cross_entropy(logits,letters[target])
        margin=(scores.gather(1,target[:,None])-scores.gather(1,reject[:,None])-(ref.gather(1,target[:,None])-ref.gather(1,reject[:,None]))).squeeze(1)
        dpo=torch.nn.functional.softplus(-margin[valid]).mean() if valid.any() else scores.sum()*0
        loss=sft+.5*dpo;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();timer.cancel()
        logs.append({'step':step+1,'loss':float(loss.detach()),'seconds':time.perf_counter()-start,'tokens':inputs['input_ids'].shape[1]})
        if (step+1)%25==0:print(json.dumps({'arm':arm,**logs[-1]}),flush=True)
    model.save_pretrained(out/'adapter');(out/'training.json').write_text(json.dumps(logs,indent=2))
    model.eval();dev=load_states('dev');selected=[r for i,r in enumerate(dev) if i%5==0][:48];pred=[]
    for index in range(0,len(selected),2):
        batch=selected[index:index+2];rendered=[prompt(r['observation'],names['fresh'],seed_for(r['base_group_id'])) for r in batch]
        with torch.no_grad():scores=model_logits(model,encode(tokenizer,[x[0] for x in rendered]))[:,letters]
        for r,(text,actions),s in zip(batch,rendered,scores):
            choice=int(s[:len(actions)].argmax());good=[r['candidates'][i]['action'] for i in r['optimal_indices']]
            pred.append({'world_id':r['world_id'],'correct':actions[choice] in good,'selected':actions[choice],'scores':s.tolist()})
    (out/'dev_predictions.json').write_text(json.dumps(pred,indent=2));print(json.dumps({'dev_planner_correct':sum(x['correct'] for x in pred),'total':len(pred),'main_test_seen':False}))

if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
