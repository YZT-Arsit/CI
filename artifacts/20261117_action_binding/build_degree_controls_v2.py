"""Two equal cycles in every provider; only anchored co-membership differs."""
import copy,json,sys,hashlib,collections
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import targets
from audit_data import check

def anchored_pair(seed,cycle_length,kind):
    assert cycle_length>=4 and cycle_length%2==0
    base=make_world(seed,kind,depth=2*cycle_length-1,noise=2,mode='both',allow=False,required=True,delivered=True);base['base_group_id']=f'anchored_cycles_{seed}';swaps=[]
    for provider in base['providers']:
        nodes=[provider['ticket']]+[b for a,b in provider['links']];old_history=nodes[-1];anchor=nodes[cycle_length//2];other=nodes[cycle_length+cycle_length//2];swaps.append((anchor,other))
        for history in base['histories']:
            if history[0]==old_history:history[0]=anchor
        a=nodes[:cycle_length];b=nodes[cycle_length:];provider['links']=[list(edge) for cycle in (a,b) for edge in zip(cycle,cycle[1:]+cycle[:1])]
    result=[]
    for side in (0,1):
        world=copy.deepcopy(base);world['world_id']+=f'_side{side}';a,b=swaps[side];mapping={a:b,b:a};provider=world['providers'][side];provider['links']=[[mapping.get(x,x),mapping.get(y,y)] for x,y in provider['links']];result.append(world)
    return result

def profile(provider,histories):
    edges=provider['links'];out=dict(edges);indegree=collections.Counter(b for a,b in edges);assert len(out)==len(edges) and all(n==1 for n in indegree.values());seen=set();cycles=[]
    for start in out:
        if start in seen:continue
        at=start;component=[]
        while at not in seen:seen.add(at);component.append(at);at=out[at]
        assert at==start;cycles.append(len(component))
    keys={h[0] for h in histories};ticket=provider['ticket']
    return {'nodes':sorted(out),'indegree':dict(indegree),'outdegree':dict(collections.Counter(a for a,b in edges)),'cycle_lengths':sorted(cycles),'history_to_ticket_edge':any((h,ticket) in map(tuple,edges) for h in keys),'ticket_to_history_edge':any((ticket,h) in map(tuple,edges) for h in keys)}

def run():
    out=R/'degree_controls_v2';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();worlds=[];states=[]
    for length in (4,6,8,12):
        for n in range(4):
            a,b=anchored_pair(217000000+length*100+n,length,('meeting','query')[n%2]);assert a['histories']==b['histories']
            for pa,pb in zip(a['providers'],b['providers']):
                x=profile(pa,a['histories']);y=profile(pb,b['histories']);assert x==y and x['cycle_lengths']==[length,length] and not x['history_to_ticket_edge'] and not x['ticket_to_history_edge'] and pa['ticket']==pb['ticket']
            pair=[targets(w)[0] for w in (a,b)];assert pair[0]['optimal_indices']!=pair[1]['optimal_indices'] and all(len(r['optimal_indices'])==1 and not check(r) for r in pair);worlds.extend([a,b]);states.extend(pair)
    path=out/'dataset.json';path.write_text(json.dumps({'worlds':worlds,'states':states},ensure_ascii=False),encoding='utf-8');plan={'scope':'new DEV anchored connectivity; every provider always has two equal directed cycles','base_world_pairs':16,'nodes_per_provider':[8,12,16,24],'reachable_ticket_to_history_distance':[2,3,4,6],'same_history_nodes_degrees_counts_cycle_lengths':True,'direct_anchor_edge_cues_always_false':True,'different_unique_safe_choice':True,'independent_bfs_passed':True,'prior_control_limitation':'v1 degree controls retain a perfect H-to-ticket closing-edge shortcut; v1 remains archived and unpromoted','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'not_a_proof_of_model_learning':True};(out/'PLAN.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))

if __name__=='__main__':run()
