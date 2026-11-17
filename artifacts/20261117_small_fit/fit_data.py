"""Bounded shared-query/actor diagnostic using an outcome-independent TRAIN subset."""
import json
import random
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
PARENT = R.parent / '20261117_contrast_repair'
sys.path.insert(0, str(PARENT))
from train_probe import verify as verify_parent
from paths import read, rows, sha
from render import query


def batches():
    selected = {}
    for batch in rows(PARENT / 'train_batches.jsonl'):
        if not batch['forced_contrast']:
            continue
        key = (batch['key'][0], batch['mixed_depth'], batch['factors'][0])
        selected.setdefault(key, batch)
    assert len(selected) == 18
    return [selected[key] for key in sorted(selected)]


def schedule():
    rng = random.Random(59117)
    result = []
    for epoch in range(8):
        order = list(range(18))
        rng.shuffle(order)
        result.extend(order)
    return result


def cases():
    pools = read(R.parent / '20261116_relation_solution/entity_palettes.json')
    observations, labels = [], []
    for bi, batch in enumerate(batches()):
        depth = batch['mixed_depth']
        pair = batch['pairs'][str(depth)]
        for view, seeds in enumerate(batch['views']):
            for ci, (a, b) in enumerate(zip(pair[0]['candidates'], pair[1]['candidates'])):
                assert a['action'] == b['action']
                group = f'{bi}_{view}_{ci}'
                for vocab, palette_name in [('original', 'train'), ('renamed', 'fresh')]:
                    for side, row in enumerate(pair):
                        candidate = row['candidates'][ci]
                        identifier = f'{group}_{vocab}_{side}'
                        observations.append({'id': identifier, 'group': group, 'vocab': vocab,
                                             'side': side, 'prompt': query(row['observation'], pools[palette_name], seeds, candidate['action'])})
                        labels.append({'id': identifier, 'role': batch['key'][0], 'depth': depth,
                                       'kind': batch['factors'][0], 'target': candidate['safe_completion'],
                                       'changing': a['safe_completion'] != b['safe_completion']})
    return observations, labels


def verify():
    verify_parent()
    plan = read(R / 'PLAN.json')
    assert sha(PARENT / 'SEALED.json') == plan['parent_seal_sha256']
    assert sha(PARENT / 'TRAIN_PROBE_SEALED.json') == plan['parent_train_probe_seal_sha256']
    assert all(sha(R / name) == value for name, value in {**plan['sources'], **plan['data']}.items())
    assert all(sha(ROOT / name) == value for name, value in plan['dependencies'].items())
    return plan


def build():
    assert not (R / 'PLAN.json').exists()
    verify_parent()
    bs = batches()
    obs, labels = cases()
    assert len(obs) == 336 and len({x['group'] for x in obs}) == 84
    assert sum(x['changing'] for x in labels) == 192
    for name, data in [('batches.jsonl', bs), ('observations.jsonl', obs), ('labels.jsonl', labels)]:
        (R / name).write_text(''.join(json.dumps(x) + '\n' for x in data), encoding='utf-8')
    cp = PARENT / 'repair/mixed_depth'
    deps = read(PARENT / 'PLAN.json')['dependencies'].copy()
    for folder in [PARENT, R.parent / '20261117_teacher_binding', R.parent / '20261117_frozen_crossover']:
        for p in folder.glob('*.py'):
            deps[p.relative_to(ROOT).as_posix()] = sha(p)
    for name in ['train_batches.jsonl', 'fresh_observations.jsonl', 'fresh_labels.jsonl',
                 'repair/mixed_depth/fresh_raw.jsonl', 'TRAIN_PROBE_PLAN.json', 'PLAN.json']:
        p = PARENT / name
        deps[p.relative_to(ROOT).as_posix()] = sha(p)
    plan = {'arms': ['initial', 'query_only', 'joint'], 'steps': 144, 'schedule': schedule(),
            'lr': .0002, 'seed': 17, 'checkpoint': {'path': cp.relative_to(ROOT).as_posix(), 'seal': read(cp / 'CHECKPOINT_SEALED.json')},
            'parent_seal_sha256': sha(PARENT / 'SEALED.json'),
            'parent_train_probe_seal_sha256': sha(PARENT / 'TRAIN_PROBE_SEALED.json'),
            'sources': {p.name: sha(p) for p in R.glob('*.py')},
            'data': {name: sha(R / name) for name in ['batches.jsonl', 'observations.jsonl', 'labels.jsonl', 'PLAN.md']},
            'dependencies': deps, 'query_responses_per_arm': 336, 'new_oracle_responses_per_trained_arm': 144,
            'scope': 'TRAIN fit and same-world renaming only. No independent generalization, complete autonomous policy or new natural-norm claim.',
            'full_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'selected_pairs': len(bs), 'steps_per_trained_arm': 144, 'query_responses_per_arm': len(obs),
                      'condition_cells': 18, 'changing_pairs_per_vocab': 48, 'dependencies': len(deps)}))


if __name__ == '__main__':
    build()
