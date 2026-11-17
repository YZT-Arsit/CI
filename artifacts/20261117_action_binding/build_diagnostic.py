"""Freeze grouped TRAIN/new-DEV intervention cases, without loading any model."""
import json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=ROOT/'artifacts/20261116_relation_solution';sys.path.insert(0,str(ROOT/'code'))
from experiments.relation_solution_data import training_world,intervene,targets
from experiments.receipt_dataset import make_world
from kci.dynamic_tasks import canonical

def run():
    assert not (R/'cases.jsonl').exists();worlds=[]
    for depth in (1,2,3):
        candidates=[]
        for index in range(0,1152,2):
            w=training_world(index)
            if len(w['providers'][0]['links'])!=depth:continue
            row=targets(w)[0]
            if len(row['optimal_indices'])==1 and w['mode']=='one':candidates.append(w)
        # Two replicas of each computational task at zero noise, fixed before model calls.
        selected=[]
        for kind in ('meeting','query'):selected += [w for w in candidates if w['kind']==kind][:2]
        assert len(selected)==4
        for n,w in enumerate(selected):worlds.append(('train',depth,n,w))
        for n in range(4):
            w=make_world(210000000+depth*100+n,('meeting','query')[n%2],depth=depth,noise=0,mode='one',allow=False,required=True,delivered=True)
            w['base_group_id']=w['world_id'];worlds.append(('new_dev',depth,n,w))
    rows=[]
    for split,depth,n,w in worlds:
        pair_id=f'{split}_d{depth}_{n}'
        pair=[]
        for side in (0,1):
            world=intervene(w,'swap') if side else w
            row=targets(world)[0];assert row['observation']['role']=='Planner' and len(row['optimal_indices'])==1
            row.update(pair_id=pair_id,side=side,split=split,depth=depth)
            row['optimal_action']=row['candidates'][row['optimal_indices'][0]]['action'];pair.append(row)
        assert pair[0]['optimal_action']!=pair[1]['optimal_action'];rows+=pair
    path=R/'cases.jsonl';path.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
    plan={'baseline_commit':'b476787ba9e0cebf19d6b81569eeaf36b0570fbc','models':['base','plain_17','trace_17','balanced_17'],'modes':['direct','gold_trace','swapped_trace'],'cases':len(rows),'groups':len(worlds),'depths':[1,2,3],'groups_per_split_depth':4,'nonce':7331,'cases_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'forced-prefix diagnostic only; oracle prefixes never count as autonomous success','selection':'first two qualifying replicas per kind/depth in old train; four fresh fixed-seed cases per depth','test_data_used':False,'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'code/experiments/relation_solution_data.py']}}
    (R/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in plan.items() if k!='source_sha256'}))

if __name__=='__main__':run()
