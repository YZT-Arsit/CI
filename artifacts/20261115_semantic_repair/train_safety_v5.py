"""Persistent CPU worker; setup separate, each training chunk <=9.5 seconds."""
import os,sys,time,json,threading,random
from pathlib import Path
from hashlib import sha256

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path.insert(0,str(ROOT/'.upload_work/site-packages/torch_cpu') if os.name=='nt' else
    json.loads((R/'wsl_environment.json').read_text())['private_package_path'])
sys.path.insert(0,str(ROOT/'code'))
setup=time.perf_counter()
import numpy as np
import torch
from kci.receipt_policy import GraphPolicy,TokenPolicy,graph_encode
from token_input_v3 import token_encode,augment_pairs
torch.set_num_threads(2);torch.set_num_interop_threads(1)


def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def load_rows():
    data=[]
    for p in sorted((R/'dataset/train_safety').glob('preferences_*.jsonl')):data += [json.loads(x) for x in p.read_text().splitlines()]
    worlds={r['world_id'] for r in data}
    if len(worlds)!=1024:raise ValueError(f'training group coverage incomplete: {len(worlds)}')
    return data


def train():
    rows=load_rows();out=R/'models_v5';out.mkdir(exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('frozen training run exists')
    files=[ROOT/'code/kci/receipt_tasks.py',ROOT/'code/kci/receipt_policy.py',ROOT/'code/experiments/receipt_dataset.py',Path(__file__),R/'token_input_v3.py',R/'build_safety.py']
    hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}
    models=[(kind,arm,seed) for kind in ('graph','token') for arm in (('ci_only','combined','shuffled','no_relations') if kind=='graph' else ('ci_only','combined')) for seed in (17,29,41)]
    plan={'models':models,'steps':{'graph':1200,'token':6000},'chunk_steps':5,'batch_pairs':32,'graph_lr':.004,'token_lr':.001,
          'seeded_fixed_training':True,'test_read_during_training':False,'no_gpu':True,
          'loss':'finite-action DPO with uniform reference, beta=1; softplus(-(score_chosen-score_rejected))',
          'source_hashes':hashes,'torch':torch.__version__,'torch_setup_seconds':time.perf_counter()-setup,
          'entity_token_embeddings':'fixed random codes; fresh-token evaluation uses unused rows',
          'training_rows':len(rows),'train_worlds':1024}
    save(out/'plan.json',plan);print(json.dumps({'ready':True,'models':len(models),'rows':len(rows),'torch':torch.__version__}),flush=True)
    # No hidden outcomes enter either encoder. No test records are loaded here.
    gx=[];ga=[];ids=[]
    for row in rows:
        for side in ('chosen','rejected'):
            x,a=graph_encode(row['observation'],row[side]);gx.append(x);ga.append(a)
            ids.append(token_encode(row['observation'],row[side]))
    gx=torch.tensor(np.array(gx));ga=torch.tensor(np.array(ga));ids=torch.tensor(np.array(ids))
    all_reports=[]
    for kind,arm,seed in models:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=GraphPolicy(relations=arm!='no_relations') if kind=='graph' else TokenPolicy()
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        opt=torch.optim.Adam(model.parameters(),lr=.004 if kind=='graph' else .001)
        eligible=[i for i,r in enumerate(rows) if (r['level']=='immediate' if arm=='ci_only' else r['level']=='future' if arm=='future_only' else True)]
        swap=rng.integers(0,2,len(rows)) if arm=='shuffled' else np.zeros(len(rows),int)
        folder=out/f'{kind}_{arm}_{seed}';folder.mkdir();losses=[];receipts=[]
        total_steps=1200 if kind=='graph' else 6000
        chunk_steps=5
        for chunk in range(total_steps//chunk_steps):
            def timeout():
                with (out/'timeouts.jsonl').open('a') as f:f.write(json.dumps({'model':folder.name,'chunk':chunk,'timeout':9.5})+'\n')
                os._exit(124)
            timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();t=time.perf_counter()
            try:
                for _ in range(chunk_steps):
                    buckets=([('Source',None,12),('Worker','immediate_ci',8),('Worker','task_utility',12)] if arm=='ci_only' else [('Planner',None,12),('Source',None,8),('Worker','immediate_ci',4),('Worker','task_utility',8)])
                    choice=np.concatenate([rng.choice([i for i in eligible if rows[i]['observation']['role']==role and (supervision is None or rows[i]['supervision']==supervision)],n) for role,supervision,n in buckets]);first=2*choice+swap[choice];second=2*choice+1-swap[choice]
                    index=np.concatenate([first,second])
                    scores=model(gx[index],ga[index]) if kind=='graph' else model(torch.tensor(augment_pairs(ids[index].numpy(),rng)))
                    loss=torch.nn.functional.softplus(-(scores[:32]-scores[32:])).mean()
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                    losses.append(float(loss.detach()))
                receipt={'chunk':chunk,'steps':chunk_steps,'seconds':time.perf_counter()-t,'last_loss':losses[-1]}
                receipts.append(receipt)
                # Checkpoint every completed chunk, so a timeout never erases work.
                if (chunk+1)%20==0:torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'steps':chunk_steps*(chunk+1),
                            'rng':rng.bit_generator.state},folder/'latest.pt')
            finally:timer.cancel()
            with (out/'chunks.jsonl').open('a') as f:f.write(json.dumps({'model':folder.name,**receipt})+'\n')
            print(json.dumps({'model':folder.name,**receipt}),flush=True)
        delta=float(sum((v-initial[k]).square().sum() for k,v in model.state_dict().items()))
        report={'kind':kind,'arm':arm,'seed':seed,'trainable_parameters':sum(p.numel() for p in model.parameters()),
                'training_pairs':len(eligible),'steps':total_steps,'initial_mean_loss':float(np.mean(losses[:50])),
                'final_mean_loss':float(np.mean(losses[-50:])),'parameter_squared_change':delta,
                'entity_embedding_rows_unchanged':bool(torch.equal(model.embedding.weight[64:].detach(),initial['embedding.weight'][64:])) if kind=='token' else None,
                'chunks':receipts,'test_outputs_seen':False,'checkpoint_sha256':sha256((folder/'latest.pt').read_bytes()).hexdigest()}
        save(folder/'training.json',report);all_reports.append(report)
    assert all(sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    save(out/'CHECKPOINTS_SEALED.json',{'models':all_reports,'source_unchanged':True,'test_read':False})


if __name__=='__main__':train()
