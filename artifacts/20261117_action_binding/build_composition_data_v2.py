"""Match base groups, repetition and role identities across depth curricula."""
from build_composition_data import *

def run():
    out=R/'composition_data_v2';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();train={'one_hop':[],'mixed_depth':[]};count=0
    for n in range(32):
        for arm in train:
            depth=1 if arm=='one_hop' else 2+n%2;kind=('meeting','query')[(n//2)%2];worlds=chain_pair(235000000+n,depth,kind);rows=[targets(w) for w in worlds];assert all(len(x)==5 for x in rows) and not [e for side in rows for row in side for e in check(row)];train[arm].append({'base_group_id':worlds[0]['base_group_id'],'depth':depth,'worlds':worlds,'states':rows});count+=10
    assert [g['base_group_id'] for g in train['one_hop']]==[g['base_group_id'] for g in train['mixed_depth']]
    dev=json.loads((R/'composition_data/dev_worlds.json').read_text());assert not {g['base_group_id'] for g in train['one_hop']} & {w['base_group_id'] for ws in dev.values() for w in ws}
    for name,payload in [('train',train),('dev_worlds',dev)]:
        (out/(name+'.json')).write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    old=json.loads((R/'composition_data/PLAN.json').read_text());plan={'base_groups_per_arm':32,'same_base_group_ids_and_seeds':True,'same_group_sampling_and_repetition_required':True,'depths':{'one_hop':[1],'mixed_depth':[2,3]},'mixed_depth_independent_of_kind':True,'train_world_versions':128,'train_state_versions':640,'independent_training_labels_checked':count,'dev_worlds':80,'paired_conditions':old['paired_conditions'],'dev_source_sha256':hashlib.sha256((R/'composition_data/dev_worlds.json').read_bytes()).hexdigest(),'source_computation_holdout':False,'source_computation_instances':['meeting_original_instance_01','query_original_instance_01'],'scope':'matched synthetic-base-group curriculum; not source-level task holdout','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(out/'PLAN.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))

if __name__=='__main__':run()
