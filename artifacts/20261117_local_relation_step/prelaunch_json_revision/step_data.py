"""One-round relation probe; labels never enter a query except declared examples."""
import collections
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
R = Path(__file__).resolve().parent; ROOT = R.parents[1]
PARENT = R.parent / '20261117_rename_coverage'
sys.path.insert(0, str(PARENT))
from rename_data import verify as verify_parent


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def rows(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def reached_after_one(frontier, edges):
    initial = set(frontier)
    return initial | {destination for source, destination in edges if source in initial}


def pair(seed, noise):
    rng = random.Random(seed); candidates = [(a, b) for a in range(12) for b in range(12)
        if a != b and (a, b) not in [(0, 3), (2, 4), (0, 4), (2, 3), (1, 3)]]
    extra = rng.sample(candidates, noise)
    return [{'id': str(seed) + '_' + str(side), 'group': str(seed), 'frontier': [0, 1], 'target': 3,
             'edges': edges + extra, 'noise': noise, 'kind': 'switch', 'side': side}
            for side, edges in enumerate([[(0, 3), (2, 4)], [(0, 4), (2, 3)]])]


def worlds():
    train = [pair(450000000 + i, [0, 2, 6][i % 3]) for i in range(64)]
    dev = [pair(450010000 + noise * 100 + i, noise) for noise in [0, 2, 6, 12] for i in range(8)]
    controls = []
    for kind in ['known', 'empty_frontier', 'two_hop_only', 'direct']:
        for i in range(4):
            row = copy.deepcopy(pair(450020000 + len(controls), 2)[0])
            row.update(kind=kind, side=0, edges=[(0, 2), (2, 3), (4, 5), (5, 6)])
            if kind == 'known': row['frontier'] = [0, 3]
            elif kind == 'empty_frontier': row['frontier'] = []
            elif kind == 'direct': row['edges'][0] = (0, 3)
            controls.append(row)
    return train, dev, controls


def prompt(row, pool, seed):
    rng = random.Random(seed); names = rng.sample(pool, 12)
    frontier = row['frontier'][:]; edges = row['edges'][:]
    rng.shuffle(frontier); rng.shuffle(edges)
    return ('Directed edges go from the first name to the second. Start with the known nodes. '
            'Keep them and add destinations of edges whose source was known at the start. '
            'Do exactly one round; do not follow a second edge in this round.\n'
            'Known nodes: ' + (', '.join(names[x] for x in frontier) or 'none') + '.\n'
            'Edges: ' + '; '.join(names[a] + ' -> ' + names[b] for a, b in edges) + '.\n'
            'Is ' + names[row['target']] + ' known after this one round? Reply ANSWER: followed by 1 for yes or 0 for no.')


def label(row):
    # Independent finite matrix multiplication: no use of reached_after_one.
    vector = [int(i in row['frontier']) for i in range(12)]
    matrix = [[int((i, j) in row['edges']) for j in range(12)] for i in range(12)]
    return int(vector[row['target']] or sum(vector[i] * matrix[i][row['target']] for i in range(12)))


def cases():
    train, dev, controls = worlds(); pools = read(PARENT / 'palettes.json')
    demos = [r for pair_ in train[:2] for r in pair_]
    random.Random(45017).shuffle(demos)
    examples = '\n\n'.join('Example:\n' + prompt(row, pools['train'], 45017 + i) + '\nANSWER:\n' + str(label(row)) for i, row in enumerate(demos))
    output, labels = [], []
    for world in [r for p in dev for r in p] + controls:
        for vocab, key in [('familiar', 'train'), ('third', 'new')]:
            for view in [0, 1]:
                text = prompt(world, pools[key], int(world['group']) + view * 100000)
                for mode in ['zero_shot', 'four_examples']:
                    identifier = f"{world['id']}_{vocab}_{view}_{mode}"
                    output.append({'id': identifier, 'group': world['group'], 'side': world['side'], 'vocab': vocab,
                                   'view': view, 'mode': mode, 'kind': world['kind'], 'noise': world['noise'],
                                   'prompt': text if mode == 'zero_shot' else examples + '\n\nNow answer this case:\n' + text})
                    labels.append({'id': identifier, 'target': label(world), 'world': world})
    return output, labels


def verify():
    verify_parent(); plan = read(R / 'PLAN.json')
    assert sha(PARENT / 'SEALED.json') == plan['parent_seal_sha256']
    assert all(sha(R / n) == h for n, h in {**plan['sources'], **plan['data']}.items())
    assert all(sha(ROOT / n) == h for n, h in plan['dependencies'].items())
    return plan


def build():
    assert not (R / 'PLAN.json').exists(); verify_parent()
    train, dev, controls = worlds()
    for a, b in train + dev:
        assert label(a) == 1 and label(b) == 0
        for column in [0, 1]:
            assert collections.Counter(e[column] for e in a['edges']) == collections.Counter(e[column] for e in b['edges'])
        assert len(a['edges']) == len(b['edges']) and a['frontier'] == b['frontier'] and a['target'] == b['target']
    for row in [r for p in train + dev for r in p] + controls:
        assert label(row) == int(row['target'] in reached_after_one(row['frontier'], row['edges']))
    obs, labels = cases(); assert len(obs) == 640
    (R / 'worlds.json').write_text(json.dumps({'train_pairs': train, 'dev_pairs': dev, 'controls': controls}), encoding='utf-8')
    for name, values in [('observations.jsonl', obs), ('labels.jsonl', labels)]:
        (R / name).write_text(''.join(json.dumps(v) + '\n' for v in values), encoding='utf-8')
    parent_plan = read(PARENT / 'PLAN.json'); deps = parent_plan['dependencies'].copy()
    for name in ['PLAN.json', 'SEALED.json', 'palettes.json', 'rename_data.py', 'copy_sources.py', 'fresh_worlds.py', 'rename_views.py']:
        deps[(PARENT / name).relative_to(ROOT).as_posix()] = sha(PARENT / name)
    plan = {'arms': ['fixed', 'renamed'], 'new_responses': 1280, 'new_updates': 0,
            'checkpoints': {a: {'path': (PARENT / 'study' / a).relative_to(ROOT).as_posix(), 'seal': read(PARENT / 'study' / a / 'CHECKPOINT_SEALED.json')} for a in ['fixed', 'renamed']},
            'parent_seal_sha256': sha(PARENT / 'SEALED.json'), 'dependencies': deps,
            'sources': {p.name: sha(p) for p in R.glob('*.py')},
            'data': {n: sha(R / n) for n in ['PLAN.md', 'worlds.json', 'observations.jsonl', 'labels.jsonl']},
            'gate': 'For each prompt mode separately: >=95% paired correctness in each noise x vocabulary cell, all controls correct, zero invalid and paired token lengths equal.',
            'scope': 'Frozen local-step interface diagnostic, not closed-loop policy or proof of absent capacity.', 'whole_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8'); print(json.dumps({'responses': 1280, 'updates': 0, 'per_model': len(obs)}))


if __name__ == '__main__': build()
