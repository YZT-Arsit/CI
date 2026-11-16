"""DEV-motivated component-balanced proof SFT, new frozen three-seed run."""
from lm_common import *
import time,threading,hashlib,re
from proof_targets import response

def pairs():
    indexed={}
    for row in load_states('train'):
        obs=row['observation'];key=(row['base_group_id'],obs['role'],obs['committed_ticket'])
        indexed.setdefault(key,[]).append(row)
    buckets={}
    for rows in indexed.values():
        assert len(rows)==2
        if all(len(r['optimal_indices'])==len(r['candidates']) for r in rows):continue
        key=(rows[0]['observation']['role'],any(not c['immediate_safe'] for r in rows for c in r['candidates']),any(c['safe_completion'] for r in rows for c in r['candidates']))
        buckets.setdefault(key,[]).append(sorted(rows,key=lambda r:r['world_id']))
    return buckets

def run():
    out=R/'balanced_language_models';out.mkdir(exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('immutable run exists')
    source_files=[Path(__file__),R/'proof_targets.py',R/'lm_common.py',ROOT/'code/kci/relation_language.py',ROOT/'code/kci/relation_solution.py']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    plan={'arms':['trace'],'seeds':[17,29,41],'steps':600,'lr':.0008,'batch_pairs':1,'source_hashes':hashes,'test_opened':False,
        'motivation':'unique-choice DEV showed hallucinated paths and reversed SAFE/TASK bits; no main output seen',
        'loss_weights':{'action':1.,'safe':.5,'task':.5,'path':.2,'format_including_eos':.1},'initialization':'same original base, not prior adapters'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));buckets=pairs();keys=sorted(buckets);reports=[]
    for arm in plan['arms']:
        for seed in plan['seeds']:
            tokenizer,model=setup(seed);names=palette(tokenizer);letters=answer_ids(tokenizer);rng=random.Random(seed)
            model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0008,weight_decay=0.)
            folder=out/f'balanced_{seed}';folder.mkdir();logs=[];train_tokens=0
            for step in range(plan['steps']):
                def timeout():
                    with (out/'timeouts.jsonl').open('a') as f:f.write(json.dumps({'model':folder.name,'step':step})+'\n')
                    os._exit(124)
                timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();t=time.perf_counter()
                selected=rng.choice(buckets[rng.choice(keys)]);nonce=rng.randrange(10**9)
                seq=[];labels=[];goodsets=[];response_lengths=[];components=[]
                for row in selected:
                    text,actions=prompt(row['observation'],names['train'],nonce)
                    text+=('\nGive each option SAFE (immediate), TASK (safe completion possible), PATH (may include mandatory future links), then ANSWER:\nletter.' if arm=='trace' else '\nReply ANSWER:\nletter.')
                    prefix=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
                    target,good=response(row,actions,names['train'],nonce,arm,rng)
                    tokenized=tokenizer(target+tokenizer.eos_token,add_special_tokens=False,return_offsets_mapping=True);suffix=tokenized['input_ids']
                    spans=[]
                    for match in re.finditer(r'SAFE ([01]) TASK ([01]) PATH ([^;\n]+);',target):
                        spans += [('safe',match.span(1)),('task',match.span(2)),('path',match.span(3))]
                    tags=['format_including_eos']*len(suffix)
                    for k,(left,right) in enumerate(tokenized['offset_mapping']):
                        for name,(begin,end) in spans:
                            if left<end and right>begin:tags[k]=name;break
                    tags[-2]='action';components.append(tags)
                    assert suffix[-2] in letters.tolist() and suffix[-1]==tokenizer.eos_token_id
                    seq.append(prefix+suffix);labels.append([-100]*len(prefix)+suffix);goodsets.append(good);response_lengths.append(len(suffix));train_tokens+=len(suffix)
                maximum=max(map(len,seq));keep=max(response_lengths)+1
                if maximum>1024:raise ValueError('training sequence over declared capacity')
                ids=torch.tensor([[tokenizer.pad_token_id]*(maximum-len(s))+s for s in seq]);mask=torch.tensor([[0]*(maximum-len(s))+[1]*len(s) for s in seq]);truth=torch.tensor([[-100]*(maximum-len(s))+s for s in labels])
                logits=model(input_ids=ids,attention_mask=mask,logits_to_keep=keep,use_cache=False).logits
                scores=logits[:,:-1];targets=truth[:,-keep+1:]
                loss_per=torch.nn.functional.cross_entropy(scores.reshape(-1,scores.shape[-1]),targets.reshape(-1),ignore_index=-100,reduction='none').reshape(targets.shape)
                for i,good in enumerate(goodsets):
                    last=scores[i,-2];loss_per[i,-2]=torch.logsumexp(last,0)-torch.logsumexp(last[letters[good]],0)
                pieces=[]
                for i,tags in enumerate(components):
                    offset=targets.shape[1]-len(tags);weighted=0.
                    for name,weight in plan['loss_weights'].items():
                        indices=[offset+k for k,tag in enumerate(tags) if tag==name]
                        assert indices,(name,tags)
                        weighted=weighted+weight*loss_per[i,indices].mean()
                    pieces.append(weighted)
                loss=torch.stack(pieces).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();timer.cancel()
                item={'step':step+1,'loss':float(loss.detach()),'seconds':time.perf_counter()-t,'tokens':maximum,'target_tokens':sum(response_lengths)};logs.append(item)
                if (step+1)%25==0:print(json.dumps({'model':folder.name,**item}),flush=True)
                if (step+1)%100==0:model.save_pretrained(folder/'adapter')
            model.save_pretrained(folder/'adapter');(folder/'training.json').write_text(json.dumps(logs,indent=2))
            report={'model':folder.name,'arm':arm,'seed':seed,'steps':plan['steps'],'target_tokens':train_tokens,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'test_seen':False,
                'adapter_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'adapter').iterdir() if p.is_file()}}
            (folder/'sealed.json').write_text(json.dumps(report,indent=2));reports.append(report)
            del model,opt;import gc;gc.collect()
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (out/'CHECKPOINTS_SEALED.json').write_text(json.dumps({'models':reports,'source_unchanged':True,'test_seen':False},indent=2))

if __name__=='__main__':run()
