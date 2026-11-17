"""Enumerate matched native states; save observations separately from labels."""
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(R.parent / '20261116_relation_solution')]
from experiments.relation_solution_data import targets
from audit_data import check


def main():
    assert not (R / 'PLAN.json').exists()
    source = R.parent / '20261117_binding_factors/repair/dev_worlds.json'
    prior = json.loads((source.parent / 'PLAN.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == prior['dev_sha256']
    worlds = json.loads(source.read_text())
    cases, labels = [], []
    for condition, items in worlds.items():
        for world in items:
            for row in targets(world):
                assert not check(row)
                case_id = hashlib.sha256((condition + ':' + row['state_id']).encode()).hexdigest()
                cases.append({'case_id': case_id, 'condition': condition,
                              'state_id': row['state_id'], 'world_id': row['world_id'],
                              'base_group_id': row['base_group_id'], 'observation': row['observation']})
                labels.append({'case_id': case_id, **row})
    assert len(cases) == len({r['case_id'] for r in cases}) == 560
    assert len({r['state_id'] for r in cases}) == 280
    for name, rows in [('observations', cases), ('labels', labels)]:
        (R / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    plan = {'arms': ['two_view_sft', 'conditional_js'], 'state_versions': len(cases),
            'unique_observation_states': 280, 'semantic_worlds': 56, 'base_groups': 36,
            'same_DEV_reused': True, 'source_holdout': False, 'model_updates': 0,
            'source_dev_sha256': prior['dev_sha256'],
            'files': {name: hashlib.sha256((R / name).read_bytes()).hexdigest()
                      for name in ['observations.jsonl', 'labels.jsonl', 'build.py']}}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan))


if __name__ == '__main__':
    main()
