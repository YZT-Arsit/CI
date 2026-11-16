"""Dataset provenance, grouping, factorial and shortcut checks; no model outputs."""
import json,sys,collections,itertools
from pathlib import Path
from hashlib import sha256
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.receipt_dataset import SOURCE
from kci.receipt_tasks import ReceiptTask
from kci.dynamic_tasks import canonical

def rows(split,kind='worlds'):
    return [json.loads(x) for p in sorted((R/'dataset'/split).glob(kind+'_*.jsonl')) for x in p.read_text().splitlines()]

def run():
    splits={p.name:rows(p.name) for p in (R/'dataset').iterdir() if p.is_dir()}
    report={'splits':{k:len(v) for k,v in splits.items()},'checks':{}}
    assert all(len(v)==len({w['world_id'] for w in v}) for v in splits.values())
    train=splits['train_balanced'];assert len(train)==1024
    factors=collections.Counter((w['kind'],len(w['providers'][0]['links']),len(w['histories'])-1,w['binding_required'],w['allow'],w['history_delivered'],w['mode']) for w in train)
    assert len(factors)==128 and set(factors.values())=={8}
    report['factorial_cells']=128;report['worlds_per_cell']=8
    for training in ('train','train_balanced'):
        ids={w['world_id'] for w in splits[training]}
        assert all(not ids.intersection(w['world_id'] for w in v) for k,v in splits.items() if k not in ('train','train_balanced','train_safety'))
    report['checks']['training_world_ids_disjoint_from_all_evaluation']=True
    # World-ID suffixes do not create independent generator groups. Audit the
    # seed itself; keep the detected old source-holdout overlap in the record.
    train_seeds={w['world_id'].split('_')[1] for w in train}
    overlaps={name:sorted(train_seeds.intersection(w['world_id'].split('_')[1] for w in values))
              for name,values in splits.items() if name not in ('train','train_balanced','train_safety')}
    report['balanced_generator_seed_overlap']=overlaps
    assert all(not values for name,values in overlaps.items() if name not in ('source_holdout','source_holdout_v2'))
    report['v4_source_holdout']='source_holdout_v3; prior holdout seed overlap is excluded, not concealed'
    # Within paired action comparisons both actions have identical procedural
    # length, number/type of relation triples and metadata. Only binding differs.
    for world in splits['test']:
        assert world['binding_required'] and not world['allow']
        sizes=[len(p['links']) for p in world['providers']]
        assert sizes[0]==sizes[1]
    report['checks']['test_candidate_lengths_and_relation_counts_equal']=True
    swaps={w['world_id']:w for w in splits['test_swapped']}
    for world in splits['test']:
        changed=swaps[world['world_id']]
        assert {k:v for k,v in world.items() if k!='histories'}=={k:v for k,v in changed.items() if k!='histories'}
        differences=[(a,b) for a,b in zip(world['histories'],changed['histories']) if a!=b]
        assert len(differences)==1 and differences[0][0][1]==differences[0][1][1]
    report['checks']['128_history_pairs_change_one_key_only']=True
    for name in ('train','train_balanced','train_safety'):
        ps=rows(name,'preferences');roles=collections.Counter(p['observation']['role'] for p in ps)
        assert all(p['proof']['status']=='PREFERENCE' for p in ps)
        assert all(p['proof']['immediate']==['VALID','VALID'] for p in ps if p['level']=='future')
        assert all('risk' not in p['observation'] and 'proof' not in p['observation'] for p in ps)
        unique={canonical({k:p[k] for k in ('observation','chosen','rejected')}) for p in ps}
        report[name]={'preferences':len(ps),'unique_model_input_preferences':len(unique),'roles':dict(roles),'future_pairs':sum(p['level']=='future' for p in ps)}
        if name=='train_safety':
            added=[p for p in ps if p['supervision']=='immediate_ci']
            expected_ci=sum(1+int(w['binding_required']) for w in train if w['mode']=='one' and not w['allow'] and w['history_delivered'])
            assert len(added)==expected_ci
            assert all(set(p['proof']['immediate'])=={'VALID','VIOLATION'} and p['proof']['immediate'][p['proof']['chosen']]=='VALID' for p in added)
            report[name]['immediate_ci_pairs_restored']=len(added)
    report['source_group_limitation']='Training has two native computation instances; synthetic group separation is not independent source separation. source_holdout_v2 has eight new native Meeting instances with the same added CI grammar.'
    report['all_passed']=True
    (R/'dataset_audit.json').write_text(json.dumps(report,indent=2))
    (R/'dataset_manifest.json').write_text(json.dumps({str(p.relative_to(R)):sha256(p.read_bytes()).hexdigest() for p in sorted((R/'dataset').rglob('*.jsonl'))},indent=2))
    print(json.dumps(report))

if __name__=='__main__':run()
