"""Disclosed structural positive control, data-only baseline and falsification arms."""
from lm_common import *
import numpy as np,time,hashlib,threading
from kci.relation_solution import LocalRulePolicy,DenseGraphPolicy,encode,collate,graph_encode

def short(operation,ledger,label):
    timer=threading.Timer(9.5,lambda:os._exit(124));timer.daemon=True;timer.start();t=time.perf_counter()
    try:return operation()
    finally:timer.cancel();ledger.append({'operation':label,'seconds':time.perf_counter()-t})

def run():
    torch.set_num_threads(2);out=R/'structured_models';out.mkdir(exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('immutable run')
    plan={'arms':['local','no_binding','safety_only','shuffled','dense_graph'],'seeds':[17,29,41],'primitive_steps':1000,'head_steps':1200,'graph_steps':600,
        'primitive_lr':.05,'head_lr':.01,'graph_lr':.003,'batch':64,'embedding_width':16,'teacher':'explicit local relation gates and equality supervision, architectural max-min composition; NOT an LLM method',
        'source_hashes':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/kci/relation_solution.py']}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));data=[];buckets={}
    for row in load_states('train'):
        for c in row['candidates']:
            idx=len(data);data.append((row['observation'],c['action'],[c['immediate_safe'],c['safe_completion']]))
            key=(row['observation']['role'],*data[-1][2]);buckets.setdefault(key,[]).append(idx)
    keys=sorted(buckets);labels=torch.tensor([r[2] for r in data],dtype=torch.float32);reports=[]
    encoded=[];graphs=[]
    for obs,action,_ in data:
        e=encode(obs,action,salt=811);n=max(1,int(e[3].sum()));encoded.append(tuple(x[:n] if i<4 else x for i,x in enumerate(e)))
        x,a=graph_encode(obs,action,max_nodes=40);graphs.append((x,a))
    gx=torch.tensor(np.array([x for x,a in graphs]));ga=torch.tensor(np.array([a for x,a in graphs]));del graphs
    for arm in plan['arms']:
        for seed in plan['seeds']:
            torch.manual_seed(seed);rng=random.Random(seed);nrng=np.random.default_rng(seed);ledger=[]
            model=DenseGraphPolicy() if arm=='dense_graph' else LocalRulePolicy(no_binding=arm=='no_binding')
            folder=out/f'{arm}_{seed}';folder.mkdir();y=labels.clone()
            if arm=='shuffled':
                # Shuffle within role/action so relational information, not action prior, is removed.
                groups={}
                for i,(obs,a,_) in enumerate(data):groups.setdefault((obs['role'],a.split(':')[0]),[]).append(i)
                for ids in groups.values():perm=ids.copy();rng.shuffle(perm);y[ids]=labels[perm]
            if arm!='dense_graph':
                opt=torch.optim.Adam(model.core.parameters(),lr=.05)
                for step in range(1000):
                    def update_core():
                        loss=model.core.distillation_loss(nrng);opt.zero_grad();loss.backward();opt.step()
                    short(update_core,ledger,'primitive')
                for p in model.core.parameters():p.requires_grad=False
                feature=[]
                for start in range(0,len(data),64):
                    def feature_batch():
                        batch=collate(encoded[start:start+64])
                        with torch.no_grad():return torch.cat([batch[-1],model.core(*batch[:-1])],1)
                    feature.append(short(feature_batch,ledger,'cache_learned_features'))
                feature=torch.cat(feature);opt=torch.optim.Adam(model.head.parameters(),lr=.01);steps=1200
            else:opt=torch.optim.Adam(model.parameters(),lr=.003);steps=600
            losses=[]
            for step in range(steps):
                def update_head():
                    ids=[rng.choice(buckets[rng.choice(keys)]) for _ in range(64)]
                    pred=model(gx[ids],ga[ids]) if arm=='dense_graph' else model.head(feature[ids])
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(pred[:,0] if arm=='safety_only' else pred,y[ids,0] if arm=='safety_only' else y[ids]);opt.zero_grad();loss.backward();opt.step();return float(loss.detach())
                losses.append(short(update_head,ledger,'head'))
            path=folder/'model.pt';torch.save(model.state_dict(),path)
            report={'model':folder.name,'arm':arm,'seed':seed,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'final_loss':losses[-1],'max_chunk_seconds':max(r['seconds'] for r in ledger),'parameters':sum(p.numel() for p in model.parameters())}
            (folder/'training.json').write_text(json.dumps({'report':report,'losses':losses,'timings':ledger},indent=2));reports.append(report);print(json.dumps(report),flush=True)
    (out/'CHECKPOINTS_SEALED.json').write_text(json.dumps({'models':reports,'test_seen':False},indent=2))

if __name__=='__main__':run()
