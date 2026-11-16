"""Dense executed action values with group-disjoint causal interventions."""
import itertools,json,random,copy
from pathlib import Path
from hashlib import sha256
from kci.receipt_tasks import ReceiptTask
from kci.ordering_receipt_tasks import OrderingReceiptTask
from kci.extended_meeting_tasks import ExtendedMeetingReceiptTask
from kci.dynamic_tasks import canonical
from experiments.receipt_dataset import make_world,SOURCE,observations

ROOT=Path(__file__).resolve().parents[2]
_templates={}

def environment(world):
    root=ROOT/world['source_root_relative'] if 'source_root_relative' in world else SOURCE
    cls={'ordering':OrderingReceiptTask,'extended':ExtendedMeetingReceiptTask}.get(world.get('kernel_variant'),ReceiptTask)
    key=(world['kind'],str(root),cls)
    if key not in _templates:_templates[key]=cls(world,root)
    env=copy.deepcopy(_templates[key]);env.world=copy.deepcopy(world);env.binding['world']=copy.deepcopy(world);return env

def intervene(world,mode):
    w=copy.deepcopy(world)
    if mode=='swap':
        endpoints=[p['links'][-1][1] for p in w['providers']]
        for h in w['histories']:
            if h[0] in endpoints:h[0]=endpoints[1-endpoints.index(h[0])]
    elif mode in ('broken','reverse'):
        for i,p in enumerate(w['providers']):
            index=len(p['links'])//2
            if mode=='reverse':p['links'][index].reverse()
            else:p['links'][index][1]='unrelated_'+w['world_id']+'_'+str(i)
    elif mode in ('branch','cycle'):
        for i,p in enumerate(w['providers']):
            if mode=='cycle':p['links'].append([p['links'][-1][1],p['links'][0][0]])
            else:p['links'].append([p['links'][0][0],'decoy_'+w['world_id']+'_'+str(i)])
    return w

def training_world(index):
    grid=list(itertools.product(('meeting','query'),(1,2,3),(0,2),('one','none','both'),(False,True),(False,True),(False,True)))
    cell=index//4;replica=(index%4)//2;side=index%2
    kind,depth,noise,mode,allow,required,delivered=grid[cell]
    seed=1000000+cell*2+replica
    w=make_world(seed,kind,depth=depth,noise=noise,mode=mode,allow=allow,required=required,delivered=delivered)
    w['base_group_id']=f'train_{seed}';w['factor_cell']=cell;w['mode']=mode
    if side:w=intervene(w,'swap')
    w['world_id']+=f'_side{side}'
    return w

def heldout_world(split,index):
    offsets={'dev':2000000,'test':3000000,'depth4':4000000,'depth6':5000000,'depth10':6000000,
      'noise':7000000,'optional':8000000,'authorized':9000000,'draft':10000000,'both':11000000,
      'broken':12000000,'reverse':13000000,'branch':14000000,'cycle':15000000}
    kind=('meeting','query')[index%2]
    kw={'depth':(1+index//2%4 if split=='dev' else {'depth4':4,'depth6':6,'depth10':10}.get(split,2)),
        'noise':12 if split=='noise' else 2}
    if split=='optional':kw['required']=False
    if split=='authorized':kw['allow']=True
    if split=='draft':kw['delivered']=False
    if split=='both':kw['mode']='both'
    if split=='dev':kw.update(required=(index//8)%3!=1,allow=(index//16)%3==1,mode=('one','none','both')[(index//4)%3])
    w=make_world(offsets[split]+index,kind,**kw);w['base_group_id']=w['world_id'];w['split']=split
    if split in ('broken','reverse','branch','cycle'):w=intervene(w,split)
    return w

def targets(world):
    env=environment(world);rows=[]
    for state in observations(env):
        obs=env.observation(state);choices=[]
        for action in env.actions(state):
            after,audit=env.transition(state,action);assert audit['verdict']!='UNKNOWN'
            outcome=env.exact(after);assert outcome['value'] is not None
            choices.append({'action':action,'immediate_safe':int(audit['verdict']=='VALID'),
                'safe_completion':int(outcome['safe_success']),'value':outcome['value'],
                'witness':outcome['witness'],'audit':audit})
        best=max(tuple(c['value']) for c in choices)
        rows.append({'world_id':world['world_id'],'base_group_id':world['base_group_id'],
            'state_id':sha256(canonical(obs).encode()).hexdigest(),'observation':obs,
            'candidates':choices,'optimal_indices':[i for i,c in enumerate(choices) if tuple(c['value'])==best],
            'no_safe_completion':not any(c['safe_completion'] for c in choices),'source_hashes':env.kernel.source_hashes})
    return rows

def build(output,split,start,count):
    out=Path(output)/split;out.mkdir(parents=True,exist_ok=True);worlds=[];states=[]
    for i in range(start,start+count):
        w=training_world(i) if split=='train' else heldout_world(split,i);worlds.append(w);states.extend(targets(w))
    for name,rows in [('worlds',worlds),('states',states)]:
        path=out/f'{name}_{start:04d}.jsonl'
        if path.exists():raise FileExistsError(path)
        path.write_text(''.join(canonical(x)+'\n' for x in rows),encoding='utf-8')
    print(json.dumps({'split':split,'worlds':len(worlds),'states':len(states)}))

if __name__=='__main__':
    import sys
    build(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
