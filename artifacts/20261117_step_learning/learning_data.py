"""Balanced local-operator supervision and new seed-defined confirmation groups."""
import collections
import copy
import json
import random
import sys
from pathlib import Path
R = Path(__file__).resolve().parent; ROOT = R.parents[1]
PARENT = R.parent / '20261117_local_relation_step'
NAMES = R.parent / '20261117_rename_coverage'
sys.path.insert(0, str(PARENT))
from step_data import worlds, pair, prompt, label, read, rows, sha, cases as probe_cases, verify as verify_parent


def train_worlds():
    pairs, _, controls = worlds(); result = {r['id']: r for p in pairs for r in p}
    for kind in ['known', 'empty_frontier', 'two_hop_only', 'direct']:
        source = next(r for r in controls if r['kind'] == kind)
        for i in range(8):
            r = copy.deepcopy(source); identifier = str(450030000 + len(result) - 128)
            r.update(group=identifier, id=identifier + '_0'); result[r['id']] = r
    assert len(result) == 160
    return result


def training_schedule(seed):
    data = train_worlds(); pairs, _, _ = worlds(); rng = random.Random(seed); records = []
    keys = [(f, n) for f in ['rewire', 'reverse_cycle'] for n in [0, 2, 6]] + [('control', 0), ('control', 1)]
    for epoch in range(18):
        order = keys[:]; rng.shuffle(order)
        for family, noise in order:
            if family == 'control':
                selected = [rng.choice([r for r in data.values() if r['kind'] == kind]) for kind in ['known', 'empty_frontier', 'two_hop_only', 'direct']]
                names = [rng.randrange(10**9) for _ in selected]
            else:
                selected_pairs = rng.sample([p for p in pairs if p[0]['family'] == family and p[0]['noise'] == noise], 2)
                selected = [r for p in selected_pairs for r in p]
                names = [v for _ in selected_pairs for v in [rng.randrange(10**9)] * 2]
            records.append({'step': len(records), 'kind': 'control' if family == 'control' else 'switch',
                            'stratum': [family, noise], 'ids': [r['id'] for r in selected], 'names': names,
                            'labels': [label(r) for r in selected]})
    return records


def confirmation_worlds():
    result = [r for fi, family in enumerate(['rewire', 'reverse_cycle']) for noise in [0, 2, 6, 20]
              for i in range(4) for r in pair(460010000 + fi * 10000 + noise * 100 + i, noise, family)]
    for i, row in enumerate(worlds()[2]):
        r = copy.deepcopy(row); identifier = str(460040000 + i); r.update(group=identifier, id=identifier + '_0'); result.append(r)
    return result


def confirmation_cases():
    pools = read(NAMES / 'palettes.json'); old = probe_cases()[0]
    prefixes = {o['prompt'].split('\n\nNow answer this case:\n')[0] for o in old if o['mode'] == 'four_examples'}
    assert len(prefixes) == 1; examples = next(iter(prefixes)); obs, labels = [], []
    for world in confirmation_worlds():
        for vocab, key in [('familiar', 'train'), ('third', 'new')]:
            for view in [0, 1]:
                text = prompt(world, pools[key], int(world['group']) + view * 100000)
                for mode in ['zero_shot', 'four_examples']:
                    identifier = f"{world['id']}_{vocab}_{view}_{mode}"
                    obs.append({'id': identifier, 'group': world['group'], 'side': world['side'], 'vocab': vocab, 'view': view,
                                'mode': mode, 'kind': world['kind'], 'family': world['family'], 'noise': world['noise'],
                                'prompt': text if mode == 'zero_shot' else examples + '\n\nNow answer this case:\n' + text})
                    labels.append({'id': identifier, 'target': label(world), 'world': world})
    return obs, labels


def verify():
    verify_parent(); plan = read(R / 'PLAN.json')
    assert sha(PARENT / 'PROBE_SEALED.json') == plan['parent_probe_seal_sha256']
    assert all(sha(R / n) == h for n, h in {**plan['sources'], **plan['data']}.items())
    assert all(sha(ROOT / n) == h for n, h in plan['dependencies'].items())
    return plan


def build():
    assert not (R / 'PLAN.json').exists(); verify_parent(); seal = read(PARENT / 'PROBE_SEALED.json')
    assert seal['complete'] and all(sha(PARENT / n) == h for n, h in seal['files'].items())
    latest = [r for r in read(PARENT / 'PROBE_RESULTS.json')['summaries'] if r['arm'] == 'renamed']
    assert len(latest) == 2 and not any(r['local_gate_passed'] for r in latest)
    data = train_worlds(); obs, labels = confirmation_cases(); assert len(obs) == 640
    assert not set(data) & {r['id'] for r in confirmation_worlds()}
    old_test_ids = {r['id'] for p in worlds()[1] for r in p} | {r['id'] for r in worlds()[2]}
    assert not old_test_ids & {r['id'] for r in confirmation_worlds()}
    (R / 'training_worlds.json').write_text(json.dumps(data), encoding='utf-8')
    for name, values in [('observations.jsonl', obs), ('labels.jsonl', labels)] + [(f'schedule_{s}.jsonl', training_schedule(s)) for s in [17, 29, 41]]:
        (R / name).write_text(''.join(json.dumps(v) + '\n' for v in values), encoding='utf-8')
    deps = read(PARENT / 'PLAN.json')['dependencies'].copy()
    for p in PARENT.glob('*.py'): deps[p.relative_to(ROOT).as_posix()] = sha(p)
    for name in ['PLAN.json', 'PROBE_SEALED.json', 'PROBE_RESULTS.json', 'LEARNING_DESIGN.md']:
        deps[(PARENT / name).relative_to(ROOT).as_posix()] = sha(PARENT / name)
    for name in ['study/renamed/oracle_raw.jsonl', 'SEALED.json']:
        deps[(NAMES / name).relative_to(ROOT).as_posix()] = sha(NAMES / name)
    source = NAMES / 'study/renamed'
    plan = {'arms': ['initial', 'seed17', 'seed29', 'seed41'], 'training_seeds': [17, 29, 41], 'steps': 144, 'lr': .0002,
            'checkpoint': {'path': source.relative_to(ROOT).as_posix(), 'seal': read(source / 'CHECKPOINT_SEALED.json')},
            'parent_probe_seal_sha256': sha(PARENT / 'PROBE_SEALED.json'), 'dependencies': deps,
            'sources': {p.name: sha(p) for p in R.glob('*.py')},
            'data': {n: sha(R / n) for n in ['PLAN.md', 'training_worlds.json', 'observations.jsonl', 'labels.jsonl'] + [f'schedule_{s}.jsonl' for s in [17, 29, 41]]},
            'new_updates': 432, 'new_responses': 2992, 'reused_responses': 144, 'whole_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8'); print(json.dumps({'new_updates': 432, 'new_responses': 2992, 'reused_responses': 144}))


if __name__ == '__main__': build()
