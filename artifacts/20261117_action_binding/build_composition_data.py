"""Fresh grouped chain curriculum and topology/norm DEV execution worlds."""
import json,hashlib,sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import targets,intervene
from build_structural_cases import pair as internal_pair
from build_degree_controls_v2 import anchored_pair
from audit_data import check

def chain_pair(seed,depth,kind):
    w=make_world(seed,kind,depth=depth,noise=2,mode='one',allow=False,required=True,delivered=True);w['base_group_id']=f'composition_{seed}';a=w;b=intervene(w,'swap');a['world_id']+='_side0';b['world_id']+='_side1';return [a,b]

def run():
    out=R/'composition_data';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();train={};worlds={};label_count=0
    for depth in (1,2,3):
        groups=[]
        for n in range(16):
            pair=chain_pair(230000000+depth*100+n,depth,('meeting','query')[n%2]);rows=[targets(w) for w in pair];assert all(len(x)==5 for x in rows) and not [e for side in rows for row in side for e in check(row)];label_count+=10;groups.append({'worlds':pair,'states':rows})
        train[str(depth)]=groups
    paired=['base','depth2','depth3','depth4','depth6','internal3','cycle4','cycle8']
    for index,condition in enumerate(paired):
        a=[];b=[]
        for n in range(4):
            seed=231000000+index*100+n;kind=('meeting','query')[n%2]
            if condition=='internal3':pair=internal_pair(seed,3,kind)
            elif condition.startswith('cycle'):pair=anchored_pair(seed,int(condition[5:]),kind)
            else:pair=chain_pair(seed,1 if condition=='base' else int(condition[5:]),kind)
            a.append(pair[0]);b.append(pair[1])
            rows=[targets(w) for w in pair];assert not [e for side in rows for row in side for e in check(row)];assert rows[0][0]['optimal_indices']!=rows[1][0]['optimal_indices'];label_count+=10
        worlds[condition]=a;worlds[condition+'_swap']=b
    for index,condition in enumerate(('optional','authorized','draft','both')):
        worlds[condition]=[]
        for n in range(4):
            w=make_world(232000000+index*100+n,('meeting','query')[n%2],depth=2,noise=2,mode='both' if condition=='both' else 'one',allow=condition=='authorized',required=condition!='optional',delivered=condition!='draft');w['base_group_id']=w['world_id'];worlds[condition].append(w);rows=targets(w);assert not [e for row in rows for e in check(row)];label_count+=len(rows)
    assert not {w['base_group_id'] for groups in train.values() for g in groups for w in g['worlds']} & {w['base_group_id'] for ws in worlds.values() for w in ws}
    for name,payload in [('train',train),('dev_worlds',worlds)]:
        (out/(name+'.json')).write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    plan={'train_chain_depths':[1,2,3],'train_pairs_per_depth':16,'train_worlds':96,'train_states':480,'dev_worlds':sum(map(len,worlds.values())),'paired_conditions':paired,'norm_control_depth':2,'independent_label_states_checked':label_count,'train_dev_groups_disjoint':True,'internal_or_cycle_training':False,'required_renderer':'kci.relation_language.prompt with edge/fact/option shuffling and fresh entity aliases','scope':'new DEV chain-length curriculum, with unseen topology families; no global proof or three-seed claim','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(out/'PLAN.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))

if __name__=='__main__':run()
