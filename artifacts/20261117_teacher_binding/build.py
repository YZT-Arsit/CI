"""Freeze teacher-estimate ablations with unchanged sampler and new DEV worlds."""
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(R.parent / '20261116_relation_solution'), str(R.parent / '20261117_action_binding')]
sys.path.insert(0, str(R.parent / '20261117_sampler_repair'))
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
            seed = 330000000 + {'base': 0, 'depth3': 1000, 'internal3': 2000, 'cycle4': 3000}[family] + n
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
            seed = 330010000 + {'optional': 0, 'authorized': 1000, 'draft': 2000, 'both': 3000}[condition] + n
            w = make_world(seed, ('meeting', 'query')[n % 2], depth=1, noise=2, mode='both' if condition == 'both' else 'one', allow=condition == 'authorized', required=condition != 'optional', delivered=condition != 'draft')
            for view in (0, 1):
                v = copy.deepcopy(w)
                v['base_group_id'] = v['world_id'] + '_view' + str(view)
                result[condition + '_v' + str(view)].append(v)
    return result, paired



def main():
    assert not (R / 'PLAN.json').exists()
    prior = R.parent / '20261117_sampler_repair'
    old = json.loads((prior / 'PLAN.json').read_text())
    worlds, paired = dev_worlds()
    previous = json.loads((prior / 'dev_worlds.json').read_text())
    assert not {w['world_id'] for ws in worlds.values() for w in ws} & {w['world_id'] for ws in previous.values() for w in ws}
    checked = 0
    for ws in worlds.values():
        for w in ws:
            for row in targets(w):
                assert not check(row)
                checked += 1
    for condition in paired:
        for a, b in zip(worlds[condition], worlds[condition + '_swap']):
            assert strict_reversal([targets(a)[0], targets(b)[0]])
    for condition, ws in worlds.items():
        if '_v0' in condition:
            for a, b in zip(ws, worlds[condition.replace('_v0', '_v1')]):
                a, b = copy.deepcopy(a), copy.deepcopy(b)
                a.pop('base_group_id'); b.pop('base_group_id')
                assert a == b
    (R / 'dev_worlds.json').write_text(json.dumps(worlds), encoding='utf-8')
    plan = {k: old[k] for k in ['seed', 'lr', 'batches', 'initial_adapter_hashes', 'training_files']}
    plan.update(arms=['own_sft', 'teacher_sft', 'own_query_js', 'teacher_query_js', 'earlier_sft'],
                primary_arms=['own_sft', 'teacher_sft', 'own_query_js', 'teacher_query_js'],
                evidence_arms=['own_sft', 'teacher_sft', 'own_query_js', 'teacher_query_js', 'earlier_sft'],
                baseline_checkpoints={
                    'own_sft': 'artifacts/20261117_sampler_repair/repair/repaired_sft',
                    'own_query_js': 'artifacts/20261117_value_consistency/repair/query_js',
                    'earlier_sft': 'artifacts/20261117_binding_factors/repair/two_view_sft'},
                query_weights={'teacher_sft': 0., 'teacher_query_js': 1.},
                new_steps_per_arm=80, new_total_updates=160, paired_conditions=paired,
                source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')},
                prior_plan_sha256=hashlib.sha256((prior / 'PLAN.json').read_bytes()).hexdigest(),
                old_loss_sha256=hashlib.sha256((R.parent / '20261117_value_consistency/value_loss.py').read_bytes()).hexdigest(),
                dev_sha256=hashlib.sha256((R / 'dev_worlds.json').read_bytes()).hexdigest(),
                state_versions_independently_checked=checked,
                scope='Single-seed teacher-input2x2 plus stronger reference; same TRAIN batches/source tasks, fresh DEV, no GPU/source-holdout claim.',
                actor_training_teacher_labels=True, actor_evaluation_teacher_labels=False,
                primary='Joint strict success and CI across all families; base all-four success; compare both matched controls and stronger reference.',
                oracle_at_runtime=False, whole_goal_complete=False)
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'arms': plan['arms'], 'checked': checked, 'new_updates': 160, 'dev_sha256': plan['dev_sha256']}))


if __name__ == '__main__':
    main()
