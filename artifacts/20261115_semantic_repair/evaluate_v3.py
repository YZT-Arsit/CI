"""Evaluate sealed shared policies without fallback, masks or oracle features."""
import os,sys,time,json,threading,random
from dataclasses import asdict
from pathlib import Path
from hashlib import sha256

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path.insert(0,str(ROOT/'.upload_work/site-packages/torch_cpu') if os.name=='nt' else
    json.loads((R/'wsl_environment.json').read_text())['private_package_path'])
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from kci.receipt_policy import GraphPolicy,TokenPolicy,graph_encode
from token_input_v3 import token_encode
from kci.receipt_tasks import ReceiptTask
from kci.ordering_receipt_tasks import OrderingReceiptTask
from kci.dynamic_tasks import canonical
from experiments.receipt_dataset import SOURCE,rename
torch.set_num_threads(2);torch.set_num_interop_threads(1)


def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def load_worlds(split):
    worlds=[]
    for p in sorted((R/'dataset'/split).glob('worlds_*.jsonl')):worlds += [json.loads(x) for x in p.read_text().splitlines()]
    expected={'dev':32,'test':128,'larger':64,'optional':64,'authorized':64,'undelivered':64,'both':32,
              'depth_only':64,'noise_only':64,'branching':64,
              'test_swapped':128,'test_optional_same':128,'test_authorized_same':128,'test_draft_same':128,
              'source_holdout_v2':json.loads((R/'source_qualification_v2.json').read_text())['worlds']}[split]
    if len(worlds)!=expected or len({w['world_id'] for w in worlds})!=expected:raise ValueError('incomplete/duplicate test groups: '+split)
    return worlds


def execute(model,kind,world,*,fresh=False,renamed=False):
    if renamed:world=rename(world,771)
    task=OrderingReceiptTask if world.get('kernel_variant')=='ordering' else ReceiptTask
    env=task(world,ROOT/world['source_root_relative'] if 'source_root_relative' in world else SOURCE)
    s=env.initial();trace=[]
    while not s.done:
        obs=env.observation(s);actions=list(env.actions(s))
        random.Random(world['world_id']+s.phase).shuffle(actions)
        with torch.no_grad():
            if kind=='graph':
                enc=[graph_encode(obs,a) for a in actions]
                score=model(torch.tensor(np.array([x[0] for x in enc])),torch.tensor(np.array([x[1] for x in enc])))
            else:score=model(torch.tensor(np.array([token_encode(obs,a,fresh=fresh) for a in actions])))
        choice=int(torch.argmax(score));action=actions[choice]
        after,audit=env.transition(s,action)
        trace.append({'role':obs['role'],'observation_sha256':sha256(canonical(obs).encode()).hexdigest(),
            'actions':actions,'scores':score.tolist(),'selected':action,'audit':audit})
        if audit['verdict']=='UNKNOWN':raise AssertionError('canonical model action produced UNKNOWN')
        s=after
    return {'world_id':world['world_id'],'ci':not s.violation,'success':s.success,
            'safe_success':s.success and not s.violation,'refused':trace[-1]['selected']=='WITHHOLD',
            'trace':trace,'final':asdict(s),'policy_interventions':0}


def run():
    sealed=json.loads((R/'models_v3/CHECKPOINTS_SEALED.json').read_text());out=R/'evaluation_v3';out.mkdir(exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('evaluation already started; do not overwrite')
    conditions=[(k,False,False) for k in ('test','larger','optional','authorized','undelivered','both','depth_only','noise_only','branching','source_holdout_v2','test_swapped','test_optional_same','test_authorized_same','test_draft_same')]
    conditions += [('test',False,True),('test',True,False)]
    save(out/'plan.json',{'conditions':conditions,'models_sealed_sha256':sha256((R/'models_v3/CHECKPOINTS_SEALED.json').read_bytes()).hexdigest(),
        'fallback':False,'oracle_action_mask':False,'policy_receives_private_task_truth':False,
        'primary_gate':'all viable condition safe-success rates >=95% in each model seed; both only evaluates CI because success impossible',
        'fresh_tokens':'unused fixed-random entity embedding rows, no closure/equality features in token model'})
    save(out/'source_manifest.json',{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/kci/receipt_policy.py',ROOT/'code/kci/receipt_tasks.py',ROOT/'code/kci/ordering_receipt_tasks.py',R/'token_input_v3.py']})
    results=[]
    for report in sealed['models']:
        kind,arm,seed=report['kind'],report['arm'],report['seed'];name=f'{kind}_{arm}_{seed}'
        path=R/'models_v3'/name/'latest.pt'
        assert sha256(path.read_bytes()).hexdigest()==report['checkpoint_sha256']
        model=GraphPolicy(relations=arm!='no_relations') if kind=='graph' else TokenPolicy()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);model.eval()
        summaries=[]
        for split,fresh,renamed in conditions:
            if kind=='graph' and fresh:continue
            worlds=load_worlds(split);label=split+('_renamed' if renamed else '')+('_fresh_tokens' if fresh else '')
            folder=out/name;folder.mkdir(exist_ok=True);rows=[];receipts=[]
            for offset in range(0,len(worlds),8):
                def timeout():
                    with (out/'timeouts.jsonl').open('a') as f:f.write(json.dumps({'model':name,'condition':label,'offset':offset})+'\n')
                    os._exit(124)
                timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();t=time.perf_counter()
                try:batch=[execute(model,kind,w,fresh=fresh,renamed=renamed) for w in worlds[offset:offset+8]]
                finally:timer.cancel()
                receipts.append({'offset':offset,'episodes':len(batch),'seconds':time.perf_counter()-t})
                rows.extend(batch)
            (folder/(label+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
            summary={'condition':label,'episodes':len(rows),'ci':sum(r['ci'] for r in rows),
                'safe_success':sum(r['safe_success'] for r in rows),'refused':sum(r['refused'] for r in rows),
                'policy_interventions':0,'worker_chunks':receipts}
            summaries.append(summary);print(json.dumps({'model':name,**{k:v for k,v in summary.items() if k!='worker_chunks'}}),flush=True)
        result={'model':name,'kind':kind,'arm':arm,'seed':seed,'conditions':summaries,
                'gate':all((x['ci'] if x['condition']=='both' else x['safe_success'])/x['episodes']>=.95 for x in summaries)}
        save(out/name/'summary.json',result);results.append(result)
    save(out/'EVALUATION_SEALED.json',{'models':results,'all_model_outputs_saved':True,'no_gateway':True})


if __name__=='__main__':run()
