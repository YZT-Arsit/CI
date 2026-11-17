"""Freeze a minimal schedule repair and independently check fresh DEV labels."""
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(R.parent / '20261116_relation_solution'), str(R.parent / '20261117_action_binding')]
from sampling import buckets, strict_reversal, DATA
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import intervene, targets
from build_structural_cases import pair as internal_pair
from build_degree_controls_v2 import anchored_pair
from audit_data import check


def dev_worlds():
    result, paired = {}, []
    for family, count in [('base', 8), ('depth3', 4), ('internal3', 4), ('cycle4', 4)]:
        for view in (0, 1):
            key = family + '_v' + str(view)
            result[key], result[key + '_swap'] = [], []
            paired.append(key)
        for n in range(count):
            seed = 280000000 + {'base': 0, 'depth3': 1000, 'internal3': 2000, 'cycle4': 3000}[family] + n
            kind = ('meeting', 'query')[n % 2]
            if family == 'internal3':
                pair = internal_pair(seed, 3, kind)
            elif family == 'cycle4':
                pair = anchored_pair(seed, 4, kind)
            else:
                a = make_world(seed, kind, depth=3 if family == 'depth3' else 1, noise=2, mode='one', allow=False, required=True, delivered=True)
                a['base_group_id'] = a['world_id']
                pair = [a, intervene(a, 'swap')]
            for view in (0, 1):
                for side, world in enumerate(pair):
                    w = copy.deepcopy(world)
                    w['base_group_id'] += '_view' + str(view)
                    result[family + '_v' + str(view) + ('_swap' if side else '')].append(w)
    for condition in ['optional', 'authorized', 'draft', 'both']:
        for view in (0, 1):
            result[condition + '_v' + str(view)] = []
        for n in range(4):
            seed = 280010000 + {'optional': 0, 'authorized': 1000, 'draft': 2000, 'both': 3000}[condition] + n
            w = make_world(seed, ('meeting', 'query')[n % 2], depth=1, noise=2, mode='both' if condition == 'both' else 'one', allow=condition == 'authorized', required=condition != 'optional', delivered=condition != 'draft')
            for view in (0, 1):
                v = copy.deepcopy(w)
                v['base_group_id'] = v['world_id'] + '_view' + str(view)
                result[condition + '_v' + str(view)].append(v)
    return result, paired


def main():
    assert not (R / 'PLAN.json').exists()
    old_path = R.parent / '20261117_binding_factors/repair/PLAN.json'
    old = json.loads(old_path.read_text())
    data = buckets()
    eligible = {kind: [] for kind in ['meeting', 'query']}
    for key in sorted(data):
        if key[0] != 'Planner':
            continue
        for index, pair in enumerate(data[key]):
            if strict_reversal(pair):
                eligible[pair[0]['observation']['task_kind']].append((key, index))
    rng = random.Random(28117)
    selected = {kind: rng.sample(rows, 4) for kind, rows in eligible.items()}
    batches = copy.deepcopy(old['batches'])
    changed, used_groups = [], []
    planner_steps = [i for i, b in enumerate(batches) if b['key'][0] == 'Planner']
    for n, step in enumerate(planner_steps[::2][:8]):
        kind = ('meeting', 'query')[n % 2]
        key, index = selected[kind][n // 2]
        batches[step]['key'], batches[step]['index'] = list(key), index
        changed.append(step)
        used_groups.append(data[key][index][0]['base_group_id'])
    assert len(changed) == len(set(used_groups)) == 8
    assert [i for i, (a, b) in enumerate(zip(old['batches'], batches)) if a != b] == changed
    assert all(a['views'] == b['views'] for a, b in zip(old['batches'], batches))
    for step in planner_steps:
        pair = data[tuple(batches[step]['key'])][batches[step]['index']]
        assert strict_reversal(pair) == (step in changed)
        if step not in changed:
            assert all(len(r['optimal_indices']) == len(r['candidates']) for r in pair)
    worlds, paired = dev_worlds()
    previous = json.loads((old_path.parent / 'dev_worlds.json').read_text())
    assert not {w['world_id'] for ws in worlds.values() for w in ws} & {w['world_id'] for ws in previous.values() for w in ws}
    count = 0
    for ws in worlds.values():
        for w in ws:
            for row in targets(w):
                assert not check(row)
                count += 1
    for condition in paired:
        for a, b in zip(worlds[condition], worlds[condition + '_swap']):
            assert strict_reversal([targets(a)[0], targets(b)[0]])
    for condition, ws in worlds.items():
        if '_v0' not in condition:
            continue
        for a, b in zip(ws, worlds[condition.replace('_v0', '_v1')]):
            a, b = copy.deepcopy(a), copy.deepcopy(b)
            a.pop('base_group_id'); b.pop('base_group_id')
            assert a == b
    (R / 'dev_worlds.json').write_text(json.dumps(worlds), encoding='utf-8')
    plan = {'arms': ['old_sft', 'old_js', 'repaired_sft', 'repaired_js'],
            'evidence_arms': ['old_sft', 'old_js', 'repaired_sft', 'repaired_js'],
            'baseline_checkpoints': {'old_sft': 'two_view_sft', 'old_js': 'conditional_js'},
            'new_steps_per_arm': 80, 'new_total_updates': 160, 'seed': 17, 'lr': .0008,
            'batches': batches, 'changed_steps': changed, 'strict_reversal_groups': used_groups,
            'eligible_groups_by_kind': {k: len(v) for k, v in eligible.items()},
            'planner_batches': {'strict_reversal': 8, 'all_actions_tied': 12},
            'paired_conditions': paired, 'initial_adapter_hashes': old['initial_adapter_hashes'],
            'consistency_weights': {'repaired_sft': 0., 'repaired_js': 1.},
            'old_plan_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
            'dev_sha256': hashlib.sha256((R / 'dev_worlds.json').read_bytes()).hexdigest(),
            'source_hashes': {name: hashlib.sha256((R / name).read_bytes()).hexdigest() for name in ['sampling.py', 'build.py']},
            'training_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('states_*.jsonl'))},
            'state_versions_independently_checked': count, 'source_holdout': False,
            'scope': 'Single-seed2x2CPU sampler diagnosis; same2source tasks; new DEV worlds, not sealed test',
            'oracle_at_runtime': False, 'whole_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in plan.items() if k not in ['batches', 'training_files', 'initial_adapter_hashes']}))


if __name__ == '__main__':
    main()
