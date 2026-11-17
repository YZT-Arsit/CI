"""Fixed native synthetic task transfer panel; no model-dependent selection."""
import copy
import json
import sys
from pathlib import Path
R = Path(__file__).resolve().parent; ROOT = R.parents[1]; PARENT = R.parent / '20261117_join_repair'
sys.path.insert(0, str(PARENT))
import join_data
read, rows, sha = join_data.read, join_data.rows, join_data.sha
sys.path[:0] = [str(ROOT / 'code'), str(R.parent / '20261116_relation_solution')]
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import intervene, targets, environment
from audit_data import check


def worlds():
    result = {}; paired = []
    for family, depth in [('base', 1), ('depth3', 3)]:
        for view in [0, 1]:
            key = family + '_v' + str(view); result[key] = []; result[key + '_swap'] = []; paired.append(key)
        for n, task in enumerate(['meeting', 'query']):
            seed = 670000000 + (1000 if depth == 3 else 0) + n
            a = make_world(seed, task, depth=depth, noise=2, mode='one', allow=False, required=True, delivered=True)
            a['base_group_id'] = a['world_id']; pair = [a, intervene(a, 'swap')]
            for view in [0, 1]:
                for side, world in enumerate(pair):
                    w = copy.deepcopy(world); w['base_group_id'] += '_view' + str(view)
                    result[family + '_v' + str(view) + ('_swap' if side else '')].append(w)
    for index, condition in enumerate(['optional', 'authorized', 'draft', 'both']):
        for view in [0, 1]: result[condition + '_v' + str(view)] = []
        for n, task in enumerate(['meeting', 'query']):
            w = make_world(670010000 + index * 1000 + n, task, depth=1, noise=2,
                           mode='both' if condition == 'both' else 'one', allow=condition == 'authorized',
                           required=condition != 'optional', delivered=condition != 'draft')
            for view in [0, 1]:
                v = copy.deepcopy(w); v['base_group_id'] = v['world_id'] + '_view' + str(view)
                result[condition + '_v' + str(view)].append(v)
    assert len(result) == 16 and sum(map(len, result.values())) == 32
    return result, paired


def verify():
    parent = join_data.verify(); plan = read(R / 'PLAN.json')
    assert plan['parent_plan_sha256'] == sha(PARENT / 'PLAN.json') and plan['arms'] == parent['arms']
    assert all(sha(ROOT / n) == h for n, h in plan['dependencies'].items())
    assert all(sha(R / n) == h for n, h in {**plan['sources'], **plan['data']}.items())
    ws, paired = worlds(); assert ws == read(R / 'dev_worlds.json') and paired == plan['paired_conditions']
    return plan


if __name__ == '__main__':
    assert not (R / 'PLAN.json').exists(); parent = join_data.verify(); ws, paired = worlds()
    assert not [error for cases in ws.values() for w in cases for case in targets(w) for error in check(case)]
    (R / 'dev_worlds.json').write_text(json.dumps(ws, indent=2), encoding='utf-8')
    (R / 'palettes.json').write_bytes((join_data.prior.NAMES / 'palettes.json').read_bytes())
    deps = [R.parent / '20261117_action_binding/learned_evidence.py', R.parent / '20261117_frozen_crossover/common.py',
            R.parent / '20261116_relation_solution/audit_data.py', ROOT / 'code/experiments/receipt_dataset.py', ROOT / 'code/experiments/relation_solution_data.py']
    deps += list((ROOT / 'code/kci').glob('*.py'))
    plan = {'arms': parent['arms'], 'parent_plan_sha256': sha(PARENT / 'PLAN.json'), 'paired_conditions': paired,
            'episodes_per_model': 32, 'new_episodes': 288, 'new_updates': 0, 'new_checkpoints': 0, 'oracle_at_runtime': False,
            'scope': 'Frozen original global-query policy interface, native synthetic three-phase environment, developed tasks and vocabulary. Not execution of the newly learned local-step operator, external business validation or32step collaboration.',
            'dependencies': {p.relative_to(ROOT).as_posix(): sha(p) for p in deps},
            'sources': {n: sha(R / n) for n in ['policy_data.py', 'run_policy.py', 'native_replay.py', 'audit_policy.py', 'test_policy.py', 'PLAN.md']},
            'data': {n: sha(R / n) for n in ['dev_worlds.json', 'palettes.json']}}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'episodes': 288, 'model_calls_before_freeze': 0, 'plan_sha256': sha(R / 'PLAN.json')}))
