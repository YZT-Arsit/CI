"""Reachability pairs with identical node membership and per-node in/out degrees."""
import copy,json,sys,hashlib,collections
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import targets
from audit_data import check

def degree_pair(seed,depth,kind):
    assert depth>=3 and depth%2==1
    base=make_world(seed,kind,depth=depth,noise=2,mode='both',allow=False,required=True,delivered=True);base['base_group_id']=f'degree_cycle_{seed}'
    for provider in base['providers']:
        edges=provider['links'];edges.append([edges[-1][1],provider['ticket']])
    result=[]
    for side in (0,1):
        world=copy.deepcopy(base);world['world_id']+=f'_side{side}';edges=world['providers'][side]['links'];cut=len(edges)//2-1;edges[cut][1],edges[-1][1]=edges[-1][1],edges[cut][1];result.append(world)
    return result

def degrees(provider):
    return dict(collections.Counter(a for a,b in provider['links'])),dict(collections.Counter(b for a,b in provider['links']))

def run():
    out=R/'degree_controls';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();worlds=[];states=[]
    for depth in (3,5,7,11):
        for n in range(4):
            a,b=degree_pair(216000000+depth*100+n,depth,('meeting','query')[n%2]);assert a['histories']==b['histories'] and a['subject']==b['subject']
            for pa,pb in zip(a['providers'],b['providers']):
                assert pa['ticket']==pb['ticket'] and degrees(pa)==degrees(pb) and len(pa['links'])==len(pb['links'])
                assert all(v==1 for mapping in degrees(pa) for v in mapping.values())
            pair=[targets(w)[0] for w in (a,b)];assert pair[0]['optimal_indices']!=pair[1]['optimal_indices'] and all(len(row['optimal_indices'])==1 and not check(row) for row in pair)
            worlds.extend([a,b]);states.extend(pair)
    path=out/'dataset.json';path.write_text(json.dumps({'worlds':worlds,'states':states},ensure_ascii=False),encoding='utf-8');plan={'scope':'new DEV topology diagnostic; directed cycle versus two disconnected directed cycles','base_world_pairs':16,'nodes_per_provider':[4,6,8,12],'same_history':True,'same_provider_nodes':True,'same_each_node_in_and_out_degree':True,'same_fact_counts':True,'different_unique_safe_choice':True,'independent_bfs_passed':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'not_a_proof_of_model_learning':True};(out/'PLAN.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))

if __name__=='__main__':run()
