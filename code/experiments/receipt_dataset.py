"""Group-disjoint semantic worlds and execution-derived action preferences."""
from dataclasses import asdict
from pathlib import Path
from hashlib import sha256
import argparse,json,random
from kci.dynamic_tasks import canonical
from kci.receipt_tasks import ReceiptTask,ReceiptState

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/20261113_iteration2/source_anchors'


def make_world(seed,kind='meeting',*,depth=1,noise=0,mode='one',allow=False,required=True,delivered=True):
    rng=random.Random(seed);seen=set()
    def symbol():
        while True:
            value='e'+str(rng.randrange(10**9,10**10))
            if value not in seen:seen.add(value);return value
    subject=symbol();secret=symbol();providers=[]
    for _ in range(2):
        nodes=[symbol() for _ in range(depth+1)]
        providers.append({'ticket':nodes[0],'links':[list(x) for x in zip(nodes,nodes[1:])]})
    selected=rng.randrange(2)
    keys=([providers[selected]['links'][-1][1]] if mode=='one' else
          [p['links'][-1][1] for p in providers] if mode=='both' else [symbol()])
    histories=[[key,secret] for key in keys]+[[symbol(),symbol()] for _ in range(noise)]
    rng.shuffle(histories);rng.shuffle(providers)
    return {'world_id':f'{kind}_{seed}','kind':kind,'subject':subject,'providers':providers,
            'histories':histories,'allow':allow,'binding_required':required,'history_delivered':delivered,
            'evidence_class':'source_computation_with_synthetic_receipt_obligation',
            'source_instance_group':kind+'_original_instance_01'}


def rename(world,seed):
    rng=random.Random(seed);mapping={}
    def rec(x):
        if isinstance(x,str) and x.startswith('e') and x[1:].isdigit():
            if x not in mapping:mapping[x]='z'+str(rng.randrange(10**9,10**10))
            return mapping[x]
        if isinstance(x,list):return [rec(v) for v in x]
        if isinstance(x,dict):return {k:rec(v) for k,v in x.items()}
        return x
    return rec(world)


def observations(env):
    start=env.initial();states=[start]
    for action in env.actions(start):
        after,audit=env.transition(start,action);assert audit['verdict']=='VALID'
        states.append(after);pub,audit=env.transition(after,'DELIVER_INPUTS');assert audit['verdict']=='VALID'
        states.append(pub)
    return states


def derive(world):
    env=ReceiptTask(world,SOURCE);records=[];ties=0
    for state in observations(env):
        actions=env.actions(state)
        for i,a in enumerate(actions):
          for b in actions[i+1:]:
            proof=env.preference(state,a,b)
            if proof['chosen'] is None:ties+=1;continue
            pair=[a,b];chosen=pair[proof['chosen']];rejected=pair[1-proof['chosen']]
            level='future' if env.actor(state)=='Planner' else 'immediate'
            records.append({'world_id':world['world_id'],'level':level,'observation':env.observation(state),
                'chosen':chosen,'rejected':rejected,'proof':proof,
                'source_hashes':env.kernel.source_hashes,'gpu_large_scale_eligible':False})
    return records,ties


def specification(split,index):
    # Base IDs never overlap. Semantic controls and nuisance variables are
    # determined before evaluation; no label is returned to the policy encoder.
    kind=('meeting','query')[index%2]
    base={'train':10000,'dev':20000,'test':30000,'larger':40000,'optional':50000,
          'authorized':60000,'undelivered':70000,'both':80000}[split]
    factor=index//2
    kw={'depth':1+(factor%2) if split in ('train','dev','test') else 3,
        'noise':(factor//2)%3 if split in ('train','dev','test') else 4}
    if split=='train':
        kw.update(required=factor%4!=1,allow=factor%8==3,mode='none' if factor%8==7 else 'one')
    if split=='optional':kw['required']=False
    if split=='authorized':kw['allow']=True
    if split=='undelivered':kw['delivered']=False
    if split=='both':kw['mode']='both'
    return make_world(base+index,kind,**kw)


def build(output,split,start,count):
    folder=Path(output)/split;folder.mkdir(parents=True,exist_ok=True)
    worlds=[];records=[];ties=0
    for index in range(start,start+count):
        world=specification(split,index);rows,n=derive(world);worlds.append(world);records.extend(rows);ties+=n
    key=f'{start:04d}'
    for name,data in [('worlds',worlds),('preferences',records)]:
        path=folder/f'{name}_{key}.jsonl'
        if path.exists():raise FileExistsError('fresh output required')
        path.write_text(''.join(canonical(r)+'\n' for r in data),encoding='utf-8')
    print(json.dumps({'split':split,'start':start,'worlds':len(worlds),'preferences':len(records),'ties_excluded':ties}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('split');p.add_argument('start',type=int);p.add_argument('count',type=int)
    a=p.parse_args();build(a.output,a.split,a.start,a.count)
