"""Matched local batches plus balanced original-TRAIN Worker rehearsal."""
import collections
import copy
import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path
R = Path(__file__).resolve().parent; ROOT = R.parents[1]; PARENT = R.parent / '20261117_step_learning'
sys.path.insert(0, str(PARENT))
from learning_data import NAMES, read, rows, sha, verify as verify_parent, train_worlds, training_schedule, confirmation_worlds, prompt, label, pair
sys.path.insert(0, str(ROOT / 'code'))
RENDER = R.parent / '20261117_binding_factors/render.py'
spec = importlib.util.spec_from_file_location('replay_actor_renderer', RENDER)
renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)


def replay_schedule(seed):
    pairs = worker_pairs(); pool = read(NAMES / 'palettes.json')['train']; rng = random.Random(480000000 + seed)
    canonical = pairs[0][0]['observation']['actions']; permutations = sorted(itertools.permutations(canonical)); option_seeds = {}
    for n in range(1000):
        order = canonical[:]; random.Random(n + 31).shuffle(order); option_seeds.setdefault(tuple(order), n)
        if len(option_seeds) == 6: break
    assert len(option_seeds) == 6 and all(r['observation']['actions'] == canonical for p in pairs for r in p)
    result = []
    for step in range(144):
        index = rng.randrange(len(pairs)); examples = []
        for view in [0, 1]:
            seeds = {k: rng.randrange(10**9) for k in ['names', 'facts', 'links', 'plans']}
            seeds['options'] = option_seeds[permutations[(step + view * 3) % 6]]
            for side, case in enumerate(pairs[index]):
                estimates = {c['action']: str(c['safe_completion']) for c in case['candidates']}
                text, actions = renderer.actor(case['observation'], pool, seeds, estimates)
                optimal = {case['candidates'][i]['action'] for i in case['optimal_indices']}
                good = [i for i, action in enumerate(actions) if action in optimal]
                examples.append({'id': f'{seed}_{step}_{view}_{side}', 'state_id': case['state_id'], 'pair_index': index,
                                 'view': view, 'side': side, 'seeds': seeds, 'estimates': estimates,
                                 'group': 'positive' if any(c['safe_completion'] for c in case['candidates']) else 'zero',
                                 'prompt': text, 'actions': actions, 'good_indices': good, 'answer': 'ANSWER:\n' + 'ABC'[good[0]]})
        result.append({'step': step, 'examples': examples})
    return result


def worker_pairs():
    batches = rows(NAMES / 'batches.jsonl')
    result = [b['pairs'][str(b['mixed_depth'])] for b in batches if b['pairs'][str(b['mixed_depth'])][0]['observation']['role'] == 'Worker']
    assert len(result) == 6
    return result


def confirmation_cases():
    pools = read(NAMES / 'palettes.json'); obs = []; labels = []
    for previous in confirmation_worlds():
        world = copy.deepcopy(previous); group = str(int(previous['group']) + 20000000)
        if world['kind'] == 'switch': world = pair(int(group), world['noise'], world['family'])[world['side']]
        else: world.update(group=group, id=group + '_0')
        for vocab, key in [('familiar', 'train'), ('third', 'new')]:
            for view in [0, 1]:
                identifier = f"{world['id']}_{vocab}_{view}"
                obs.append({'id': identifier, 'group': group, 'side': world['side'], 'vocab': vocab, 'view': view,
                            'kind': world['kind'], 'family': world['family'], 'noise': world['noise'],
                            'prompt': prompt(world, pools[key], int(group) + view * 100000)})
                labels.append({'id': identifier, 'target': label(world), 'world': world})
    assert len(obs) == 320
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
    obs, labels = confirmation_cases(); old = rows(PARENT / 'observations.jsonl')
    assert not {o['group'] for o in obs} & {o['group'] for o in old}
    (R / 'training_worlds.json').write_text(json.dumps(train_worlds()), encoding='utf-8')
    datasets = [('observations.jsonl', obs), ('labels.jsonl', labels)]
    for seed in [17, 29, 41]:
        datasets += [(f'local_{seed}.jsonl', training_schedule(seed)), (f'replay_{seed}.jsonl', replay_schedule(seed))]
    for name, values in datasets:
        (R / name).write_text(''.join(json.dumps(r) + '\n' for r in values), encoding='utf-8')
    deps = prior['dependencies'].copy()
    for name in [*prior['sources'], 'PLAN.json', 'SEALED.json', 'RESULTS.json']:
        deps[(PARENT / name).relative_to(ROOT).as_posix()] = sha(PARENT / name)
    for path in [NAMES / 'batches.jsonl', RENDER]: deps[path.relative_to(ROOT).as_posix()] = sha(path)
    checkpoints = {}
    for arm, folder in [('initial', 'initial')] + [(f'local{s}', f'seed{s}') for s in [17, 29, 41]]:
        path = PARENT / 'learning' / folder
        checkpoints[arm] = {'path': path.relative_to(ROOT).as_posix(), 'seal': read(path / 'CHECKPOINT_SEALED.json')}
    plan = {'arms': list(checkpoints) + ['replay17', 'replay29', 'replay41'], 'training_seeds': [17, 29, 41],
            'checkpoints': checkpoints, 'steps': 144, 'lr': .0002, 'actor_weight': 1.,
            'parent_seal_sha256': sha(PARENT / 'SEALED.json'), 'dependencies': deps,
            'sources': {p.name: sha(p) for p in R.glob('*.py')},
            'data': {n: sha(R / n) for n in ['PLAN.md', 'training_worlds.json'] + [n for n, _ in datasets]},
            'new_updates': 432, 'new_responses': 2672, 'reused_responses': 576, 'whole_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'new_updates': 432, 'new_responses': 2672, 'reused_responses': 576}))


if __name__ == '__main__': build()
