"""Matched random versus near-miss backgrounds for the same valid joins."""
import copy
import importlib.util
import itertools
import json
import random
from pathlib import Path
R = Path(__file__).resolve().parent; ROOT = R.parents[1]; FACTORS = R.parent / '20261117_relation_factors'
spec = importlib.util.spec_from_file_location('factors_for_join_repair', FACTORS / 'factor_data.py')
factors = importlib.util.module_from_spec(spec); spec.loader.exec_module(factors)
prior = factors.prior; PARENT = factors.PARENT; read, rows, sha = factors.read, factors.rows, factors.sha


def matched(seed, family, known):
    for attempt in range(500):
        core = prior.paired(seed + attempt, family, 0)
        if core[0]['n'] == 16 and len(core[0]['frontier']) == known: break
    else: raise AssertionError('bounded structural sampling exhausted')
    target = core[0]['target']; union = set(core[0]['frontier'] + core[1]['frontier'])
    forbidden = set(map(tuple, core[0]['edges'] + core[1]['edges']))
    pool = [(u, v) for u in range(16) for v in range(16) if u != v and (u, v) not in forbidden and not (u in union and v == target)]
    ordinary = random.Random(seed + 13).sample(pool, 12)
    rng = random.Random(seed + 17)
    near = rng.sample([(u, v) for u, v in pool if u in union and v != target], 6)
    near += rng.sample([(u, v) for u, v in pool if u not in union and v == target], 6)
    result = []
    for name, extra in [('random', ordinary), ('targeted', near)]:
        worlds = copy.deepcopy(core)
        for w in worlds: w['edges'] += [list(e) for e in extra]; w.update(noise=12, background=name)
        assert [prior.truth(w) for w in worlds] == [1, 0]
        assert [prior.witnesses(w) for w in worlds] == [prior.witnesses(w) for w in core]
        result.append(worlds)
    return result


def training(seed):
    rng = random.Random(seed); replay = prior.replay_schedule(seed); data = {'random': {}, 'targeted': {}}; schedule = []
    strata = [(f, k) for f in ['edge_swap', 'frontier_swap', 'direction_cycle'] for k in [1, 4]] + [('control', 0), ('control', 12)]
    for epoch in range(12):
        order = strata[:]; rng.shuffle(order)
        for family, level in order:
            step = len(schedule); base = 600000000 + seed * 1000000 + step * 1000
            if family == 'control':
                items = [prior.control(base + i, kind, level) for i, kind in enumerate(['known', 'empty', 'two_hop', 'multiple'])]
                variants = [items, copy.deepcopy(items)]; names = [rng.randrange(10**9) for _ in items]
            else:
                pairs = [matched(base + i * 500, family, level) for i in range(2)]
                variants = [pairs[0][a] + pairs[1][a] for a in [0, 1]]
                names = [n for _ in range(2) for n in [rng.randrange(10**9)] * 2]
            ids = [w['id'] for w in variants[0]]; labels = [prior.truth(w) for w in variants[0]]
            assert labels == [prior.truth(w) for w in variants[1]] and sorted(labels) == [0, 0, 1, 1]
            for arm, worlds in zip(data, variants):
                for w in worlds: assert w['id'] not in data[arm]; data[arm][w['id']] = w
            schedule.append({'step': step, 'stratum': [family, level], 'ids': ids, 'labels': labels, 'names': names,
                             'actor': replay[step]['examples']})
    assert len(schedule) == 96 and all(len(v) == 384 for v in data.values())
    return data, schedule


def fresh_standard():
    worlds = [w for fi, family in enumerate(['edge_swap', 'frontier_swap', 'direction_cycle']) for noise in [0, 4, 12, 20]
              for i in range(3) for w in prior.paired(650000000 + fi * 10000 + noise * 100 + i, family, noise)]
    worlds += [prior.control(650100000 + ki * 100 + i, kind, [0, 4, 12, 20][i]) for ki, kind in enumerate(['known', 'empty', 'two_hop', 'multiple']) for i in range(4)]
    for cell, (n, m, k, target_label) in enumerate(itertools.product([8, 16], [4, 12], [1, 3], [0, 1])):
        selected = 0
        for attempt in range(10000):
            seed = 650200000 + cell * 10000 + attempt; rng = random.Random(seed); target = rng.randrange(n)
            frontier = rng.sample([v for v in range(n) if v != target], k)
            edges = rng.sample([(u, v) for u in range(n) for v in range(n) if u != v], m)
            w = prior.record(seed, 0, n, frontier, edges, target, 'random', m, 'random')
            if prior.truth(w) == target_label: worlds.append(w); selected += 1
            if selected == 2: break
        assert selected == 2
    obs = []; labels = []; pools = read(prior.NAMES / 'palettes.json')
    for w in worlds:
        for vocab, key in [('familiar', 'train'), ('third', 'new')]:
            for view in [0, 1]:
                identifier = f"{w['id']}_{vocab}_{view}"; text, valid = prior.render(w, pools[key], int(w['group']) + view * 100000)
                obs.append({'id': identifier, 'group': w['group'], 'side': w['side'], 'vocab': vocab, 'view': view,
                            'kind': w['kind'], 'family': w['family'], 'noise': w['noise'], 'prompt': text})
                labels.append({'id': identifier, 'target': prior.truth(w), 'valid_witnesses': valid, 'world': w})
    assert len(obs) == 480
    return obs, labels


def fresh_factors():
    obs = []; labels = []; pools = read(prior.NAMES / 'palettes.json')
    for fi, family in enumerate(['retention', 'direction', 'binding']):
        for i in range(4):
            seed = 660000000 + fi * 1000 + i
            for known in [1, 4]:
                for noise in ['none', 'irrelevant', 'near_miss']:
                    for vocab, key in [('familiar', 'train'), ('third', 'new')]:
                        for w in factors.pair(seed, family, known, noise):
                            identifier = f"{seed}_{known}_{noise}_{vocab}_{w['side']}"; prompt, valid = factors.render(w, pools[key], 'binary')
                            obs.append({'id': identifier, 'seed': seed, 'family': family, 'known': known, 'noise': noise, 'vocab': vocab, 'side': w['side'], 'prompt': prompt})
                            labels.append({'id': identifier, 'target': prior.truth(w), 'valid_witnesses': valid, 'world': w})
    assert len(obs) == 288
    return obs, labels


def verify():
    factors.verify(); plan = read(R / 'PLAN.json'); seal = read(FACTORS / 'SEALED.json')
    assert sha(FACTORS / 'SEALED.json') == plan['factor_seal_sha256'] and sha(FACTORS / 'RESULTS.json') == plan['factor_results_sha256']
    assert seal['complete'] and all(sha(FACTORS / n) == h for n, h in seal['files'].items())
    assert all(sha(R / n) == h for n, h in {**plan['sources'], **plan['data']}.items())
    return plan


def build():
    factors.verify(); assert not (R / 'PLAN.json').exists(); files = []
    for seed in [17, 29, 41]:
        data, schedule = training(seed); name = f'worlds_{seed}.json'; (R / name).write_text(json.dumps(data), encoding='utf-8'); files.append(name)
        name = f'schedule_{seed}.jsonl'; (R / name).write_text(''.join(json.dumps(b) + '\n' for b in schedule), encoding='utf-8'); files.append(name)
    for panel, generated in [('standard', fresh_standard()), ('factors', fresh_factors())]:
        for part, data in zip(['observations', 'labels'], generated):
            name = f'{panel}_{part}.jsonl'; (R / name).write_text(''.join(json.dumps(o) + '\n' for o in data), encoding='utf-8'); files.append(name)
    checkpoints = {str(s): {'path': (PARENT / 'study' / f'bit{s}').relative_to(ROOT).as_posix(),
                           'seal': read(PARENT / 'study' / f'bit{s}' / 'CHECKPOINT_SEALED.json')} for s in [17, 29, 41]}
    arms = {f'initial{s}': {'seed': s, 'kind': 'initial'} for s in [17, 29, 41]}
    arms.update({f'{kind}{s}': {'seed': s, 'kind': kind} for s in [17, 29, 41] for kind in ['random', 'targeted']})
    plan = {'arms': arms, 'checkpoints': checkpoints, 'steps': 96, 'lr': .0002,
            'factor_seal_sha256': sha(FACTORS / 'SEALED.json'), 'factor_results_sha256': sha(FACTORS / 'RESULTS.json'),
            'sources': {p.name: sha(p) for p in R.glob('*.py')}, 'data': {n: sha(R / n) for n in ['PLAN.md', *files]},
            'new_updates': 576, 'new_responses': 7776, 'reused_responses': 432, 'whole_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'updates': 576, 'new_responses': 7776, 'reused_responses': 432}))


if __name__ == '__main__': build()
