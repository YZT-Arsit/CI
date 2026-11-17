"""Pre-training independent labels, view equivalence and group overlap checks."""
from repair import *
from audit_data import check
from experiments.relation_solution_data import targets

def main():
    worlds,paired=fresh_dev();states=0;families={};label_sides={}
    for condition,items in worlds.items():
        families[condition]=len(items)
        for w in items:
            rows=targets(w);assert not [e for row in rows for e in check(row)];states+=len(rows)
    for condition in paired:
        for a,b in zip(worlds[condition],worlds[condition+'_swap']):
            assert a['base_group_id']==b['base_group_id'];ra,rb=targets(a)[0],targets(b)[0];assert len(ra['optimal_indices'])==len(rb['optimal_indices'])==1 and ra['candidates'][ra['optimal_indices'][0]]['action']!=rb['candidates'][rb['optimal_indices'][0]]['action']
    comparisons=0
    for condition,items in worlds.items():
        if '_v0' not in condition:continue
        for a,b in zip(items,worlds[condition.replace('_v0','_v1')]):
            x=copy.deepcopy(a);y=copy.deepcopy(b);x.pop('base_group_id');y.pop('base_group_id');assert x==y;comparisons+=1
    previous_ids=set()
    def collect(node):
        if isinstance(node,dict):
            if 'world_id' in node and 'providers' in node:previous_ids.add(node['world_id'])
            for child in node.values():collect(child)
        elif isinstance(node,list):
            for child in node:collect(child)
    previous=HERE.parent/'20261117_action_binding'
    for p in list(previous.glob('*/dev_worlds.json'))+[previous/'alias_confirmation/worlds.json']:collect(json.loads(p.read_text(encoding='utf-8')))
    assert not previous_ids & {w['world_id'] for ws in worlds.values() for w in ws}
    payload=json.dumps(worlds).encode();result={'world_versions':sum(families.values()),'state_versions_checked':states,'equivalent_view_world_pairs':comparisons,'prior_dev_world_id_overlap':0,'source_holdout':False,'source_instances':2,'expected_dev_sha256':hashlib.sha256(payload).hexdigest(),'conditions':families,'errors':0};(HERE/'REPAIR_DATA_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
