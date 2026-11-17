"""DEV relation controls: same endpoints/history, reversed internal edge only."""
import copy,json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import targets
from audit_data import check

def pair(seed,depth,kind):
    world=make_world(seed,kind,depth=depth,noise=2,mode='both',allow=False,required=True,delivered=True)
    world['base_group_id']=f'internal_reverse_{seed}';result=[]
    for side in (0,1):
        w=copy.deepcopy(world);w['world_id']+=f'_side{side}'
        w['providers'][side]['links'][depth//2].reverse();result.append(w)
    return result

def run():
    out=R/'structural_cases';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();worlds=[];states=[]
    for depth in (3,4,6):
        for n in range(8):
            a,b=pair(213000000+depth*100+n,depth,('meeting','query')[n%2])
            assert a['histories']==b['histories']
            for pa,pb in zip(a['providers'],b['providers']):
                assert pa['ticket']==pb['ticket'] and sorted(sum(pa['links'],[]))==sorted(sum(pb['links'],[]))
                assert pa['links'][0]==pb['links'][0] and pa['links'][-1]==pb['links'][-1]
            rows=[targets(w)[0] for w in (a,b)];assert rows[0]['optimal_indices']!=rows[1]['optimal_indices']
            assert all(len(r['optimal_indices'])==1 and not check(r) for r in rows)
            worlds.extend([a,b]);states.extend(rows)
    payload={'worlds':worlds,'states':states};path=out/'dataset.json';path.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    plan={'scope':'new DEV internal-edge counterfactuals; not a final locked test','base_world_pairs':24,'depths':[3,4,6],'worlds':len(worlds),'same_pair_histories':True,'same_pair_entity_multiset':True,'same_pair_first_and_last_links':True,'different_unique_optimum':True,'independent_bfs_passed':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))

if __name__=='__main__':run()
