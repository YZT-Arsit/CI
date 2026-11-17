"""Varied one-round graphs and explicit sets of valid copied witnesses."""
import copy
import itertools
import json
import random
import sys
from pathlib import Path
R = Path(__file__).resolve().parent; ROOT = R.parents[1]; PARENT = R.parent / '20261117_step_replay'
sys.path.insert(0, str(PARENT))
from replay_data import NAMES, read, rows, sha, verify as verify_parent, replay_schedule


def witnesses(world):
    known = set(world['frontier']); target = world['target']
    return ({target} if target in known else set()) | {a for a, b in world['edges'] if a in known and b == target}


def truth(world):
    n = world['n']; vector = [int(i in world['frontier']) for i in range(n)]
    matrix = [[int([i, j] in world['edges']) for j in range(n)] for i in range(n)]
    return int(vector[world['target']] or sum(vector[i] * matrix[i][world['target']] for i in range(n)) > 0)


def options(rng):
    n = rng.choice([8, 12, 16]); k = rng.randint(1, 4)
    return n, k


def record(seed, side, n, frontier, edges, target, family, noise, kind='switch'):
    w = {'id': f'{seed}_{side}', 'group': str(seed), 'side': side, 'n': n, 'frontier': frontier,
         'edges': [list(e) for e in edges], 'target': target, 'family': family, 'noise': noise, 'kind': kind}
    assert len(set(map(tuple, w['edges']))) == len(w['edges']) and bool(witnesses(w)) == bool(truth(w))
    return w


def paired(seed, family, noise):
    rng = random.Random(seed); n, k = options(rng); a, b, target, other = rng.sample(range(n), 4)
    if family == 'direction_cycle':
        cycle = [a, target] + rng.sample([v for v in range(n) if v not in [a, target]], rng.randint(4, min(7, n - k + 1)) - 2)
        frontier = [a] + rng.sample([v for v in range(n) if v not in cycle], k - 1)
        positive = list(zip(cycle, cycle[1:] + cycle[:1])); negative = [(v, u) for u, v in positive]; f1 = frontier
    else:
        frontier = [a] + rng.sample([v for v in range(n) if v not in [a, b, target]], k - 1)
        positive = [(a, target), (b, other)]
        if family == 'edge_swap': negative = [(a, other), (b, target)]; f1 = frontier
        elif family == 'frontier_swap': negative = positive; f1 = [b] + frontier[1:]
        else: raise ValueError(family)
    forbidden = set(positive + negative); known_union = set(frontier + f1)
    candidates = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) not in forbidden and not (u in known_union and v == target)]
    extra = rng.sample(candidates, noise)
    result = [record(seed, side, n, f, e + extra, target, family, noise) for side, (f, e) in enumerate([(frontier, positive), (f1, negative)])]
    assert [truth(w) for w in result] == [1, 0]
    return result


def control(seed, kind, noise):
    rng = random.Random(seed); n, k = options(rng); a, middle, target = rng.sample(range(n), 3)
    if kind == 'known': frontier = [target] + rng.sample([v for v in range(n) if v != target], k - 1); core = []
    elif kind == 'empty': frontier = []; core = []
    elif kind == 'two_hop':
        frontier = [a] + rng.sample([v for v in range(n) if v not in [a, middle, target]], k - 1); core = [(a, middle), (middle, target)]
    elif kind == 'multiple':
        k = max(2, k); frontier = rng.sample([v for v in range(n) if v != target], k); core = [(u, target) for u in frontier[:2]]
    else: raise ValueError(kind)
    candidates = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) not in core and not (kind == 'two_hop' and u in frontier and v == target)]
    edges = core + rng.sample(candidates, noise + 4)
    w = record(seed, 0, n, frontier, edges, target, kind, noise, kind)
    assert truth(w) == int(kind in ['known', 'multiple'])
    return w


def render(world, pool, seed, task='binary'):
    rng = random.Random(seed); names = rng.sample(pool, world['n']); frontier = world['frontier'][:]; edges = world['edges'][:]
    rng.shuffle(frontier); rng.shuffle(edges)
    text = ('Directed edges go from the first name to the second. Start with the known nodes. '
            'Keep them and add destinations of edges whose source was known at the start. '
            'Do exactly one round; do not follow a second edge in this round.\n'
            'Known nodes: ' + (', '.join(names[x] for x in frontier) or 'none') + '.\n'
            'Edges: ' + '; '.join(names[a] + ' -> ' + names[b] for a, b in edges) + '.\n')
    if task == 'binary':
        text += 'Is ' + names[world['target']] + ' known after this one round? Reply ANSWER: followed by 1 for yes or 0 for no.'
    else:
        assert task == 'proof'
        text += ('Target: ' + names[world['target']] + '. Give one known node that proves the target is known after this round. '
                 'Valid choices are the target itself if already known, or any known source with an edge to the target. '
                 'If none exists, give none. Reply only PROOF: followed by that name or none.')
    valid = sorted(names[v] for v in witnesses(world)) or ['none']
    return text, valid


def training(seed):
    rng = random.Random(seed); schedule = []; worlds = {}; replay = replay_schedule(seed)
    strata = [(f, n) for f in ['edge_swap', 'frontier_swap', 'direction_cycle'] for n in [0, 4, 12]] + [('control', n) for n in [0, 4, 12]]
    for epoch in range(16):
        order = strata[:]; rng.shuffle(order)
        for family, noise in order:
            step = len(schedule); base = 500000000 + seed * 100000 + step * 10
            if family == 'control':
                items = [control(base + i, kind, noise) for i, kind in enumerate(['known', 'empty', 'two_hop', 'multiple'])]
                name_seeds = [rng.randrange(10**9) for _ in items]
            else:
                items = paired(base, family, noise) + paired(base + 1, family, noise)
                name_seeds = [v for _ in range(2) for v in [rng.randrange(10**9)] * 2]
            for w in items: assert w['id'] not in worlds; worlds[w['id']] = w
            actor_batch = copy.deepcopy(replay[step % 144])
            schedule.append({'step': step, 'stratum': [family, noise], 'ids': [w['id'] for w in items], 'names': name_seeds,
                             'labels': [truth(w) for w in items], 'replay_source_step': step % 144, 'actor': actor_batch['examples']})
    assert len(schedule) == 192 and len(worlds) == 768
    return worlds, schedule


def random_panel():
    worlds = []; cell = 0
    for n, m, k, target_label in itertools.product([8, 16], [4, 12], [1, 3], [0, 1]):
        selected = 0
        for attempt in range(10000):
            seed = 510200000 + cell * 10000 + attempt; rng = random.Random(seed); target = rng.randrange(n)
            frontier = rng.sample([v for v in range(n) if v != target], k)
            edges = rng.sample([(u, v) for u in range(n) for v in range(n) if u != v], m)
            w = record(seed, 0, n, frontier, edges, target, 'random', m, 'random')
            if truth(w) == target_label: worlds.append(w); selected += 1
            if selected == 2: break
        assert selected == 2; cell += 1
    assert len(worlds) == 32
    return worlds


def confirmation():
    worlds = [w for fi, family in enumerate(['edge_swap', 'frontier_swap', 'direction_cycle']) for noise in [0, 4, 12, 20]
              for i in range(3) for w in paired(510000000 + fi * 10000 + noise * 100 + i, family, noise)]
    worlds += [control(510100000 + ki * 100 + i, kind, [0, 4, 12, 20][i]) for ki, kind in enumerate(['known', 'empty', 'two_hop', 'multiple']) for i in range(4)]
    worlds += random_panel(); obs = []; labels = []; pools = read(NAMES / 'palettes.json')
    for world in worlds:
        for vocab, key in [('familiar', 'train'), ('third', 'new')]:
            for view in [0, 1]:
                text, valid = render(world, pools[key], int(world['group']) + view * 100000)
                proof, other = render(world, pools[key], int(world['group']) + view * 100000, 'proof'); assert valid == other
                identifier = f"{world['id']}_{vocab}_{view}"
                obs.append({'id': identifier, 'group': world['group'], 'side': world['side'], 'vocab': vocab, 'view': view,
                            'kind': world['kind'], 'family': world['family'], 'noise': world['noise'], 'prompt': text, 'proof_prompt': proof})
                labels.append({'id': identifier, 'target': truth(world), 'valid_witnesses': valid, 'world': world})
    assert len(obs) == 480
    return obs, labels


def verify():
    verify_parent(); plan = read(R / 'PLAN.json')
    assert sha(PARENT / 'SEALED.json') == plan['parent_seal_sha256']
    assert all(sha(ROOT / n) == h for n, h in plan['dependencies'].items())
    assert all(sha(R / n) == h for n, h in {**plan['sources'], **plan['data']}.items())
    return plan


def build():
    assert not (R / 'PLAN.json').exists(); verify_parent(); prior = read(PARENT / 'PLAN.json'); seal = read(PARENT / 'SEALED.json')
    assert seal['complete'] and all(sha(PARENT / n) == h for n, h in seal['files'].items())
    assert read(R / 'TOKEN_PROBE.json')['failures'] == []
    obs, labels = confirmation(); datasets = []
    for seed in [17, 29, 41]:
        worlds, schedule = training(seed); (R / f'worlds_{seed}.json').write_text(json.dumps(worlds), encoding='utf-8')
        assert not set(worlds) & {l['world']['id'] for l in labels}
        datasets.append((f'schedule_{seed}.jsonl', schedule))
    datasets += [('observations.jsonl', obs), ('labels.jsonl', labels)]
    for name, values in datasets: (R / name).write_text(''.join(json.dumps(v) + '\n' for v in values), encoding='utf-8')
    deps = prior['dependencies'].copy()
    for n in [*prior['sources'], 'PLAN.json', 'SEALED.json', 'RESULTS.json']:
        deps[(PARENT / n).relative_to(ROOT).as_posix()] = sha(PARENT / n)
    cps = {}
    for seed in [17, 29, 41]:
        path = PARENT / 'study' / f'replay{seed}'; cps[str(seed)] = {'path': path.relative_to(ROOT).as_posix(), 'seal': read(path / 'CHECKPOINT_SEALED.json')}
    arms = {f'start{s}': {'seed': s, 'kind': 'start'} for s in [17, 29, 41]}
    arms.update({f'{kind}{seed}': {'seed': seed, 'kind': kind} for seed in [17, 29, 41] for kind in ['bit', 'proof']})
    plan = {'arms': arms, 'checkpoints': cps, 'steps': 192, 'lr': .0002, 'parent_seal_sha256': sha(PARENT / 'SEALED.json'),
            'dependencies': deps, 'sources': {p.name: sha(p) for p in R.glob('*.py')},
            'data': {n: sha(R / n) for n in ['PLAN.md', 'TOKEN_PROBE.json'] + [f'worlds_{s}.json' for s in [17, 29, 41]] + [n for n, _ in datasets]},
            'new_updates': 1152, 'new_responses': 9504, 'reused_responses': 432, 'whole_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'new_updates': 1152, 'new_responses': 9504, 'reused_responses': 432}))


if __name__ == '__main__': build()
