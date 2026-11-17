"""Frozen name-only augmentation, matched panels and fresh generated DEV worlds."""
import json
import sys
from pathlib import Path
R = Path(__file__).resolve().parent
ROOT = R.parents[1]
PARENT = R.parent / '20261117_small_fit'
sys.path.insert(0, str(PARENT))
from fit_data import batches, schedule, cases, read, rows, sha, verify as verify_parent
from render import query
from rename_views import views_for
from fresh_worlds import dev_worlds, targets, check, DATA
from kci.relation_language import seed_for, vocabulary
from copy_sources import sources


def third_cases():
    pool = read(R / 'palettes.json')['new']; obs, labels = [], []
    for bi, batch in enumerate(batches()):
        depth = batch['mixed_depth']; pair = batch['pairs'][str(depth)]
        for view, seeds in enumerate(batch['views']):
            for ci, (a, b) in enumerate(zip(pair[0]['candidates'], pair[1]['candidates'])):
                assert a['action'] == b['action']
                group = f'{bi}_{view}_{ci}'
                for side, row in enumerate(pair):
                    c = row['candidates'][ci]; identifier = f'{group}_third_{side}'
                    obs.append({'id': identifier, 'group': group, 'vocab': 'third', 'side': side,
                                'prompt': query(row['observation'], pool, seeds, c['action'])})
                    labels.append({'id': identifier, 'role': batch['key'][0], 'depth': depth,
                                   'kind': batch['factors'][0], 'target': c['safe_completion'],
                                   'changing': a['safe_completion'] != b['safe_completion']})
    return obs, labels


def dev_cases(worlds):
    pool = read(R / 'palettes.json')['new']; obs, labels = [], []
    for condition, ws in worlds.items():
        for index, world in enumerate(ws):
            for si, case in enumerate(targets(world)):
                assert not check(case)
                role = case['observation']['role']; phase = {'Planner': 0, 'Source': 1, 'Worker': 2}[role]
                nonce = seed_for(world['base_group_id'], phase)
                seeds = {k: nonce for k in ['names', 'facts', 'plans', 'links', 'options']}
                for ci, c in enumerate(case['candidates']):
                    identifier = f'{condition}_{index}_{si}_{ci}'
                    obs.append({'id': identifier, 'prompt': query(case['observation'], pool, seeds, c['action'])})
                    labels.append({'id': identifier, 'condition': condition, 'world_index': index,
                                   'state_index': si, 'candidate_index': ci, 'target': c['safe_completion'], 'role': role})
    return obs, labels


def verify():
    verify_parent(); plan = read(R / 'PLAN.json')
    assert sha(PARENT / 'SEALED.json') == plan['parent_seal_sha256']
    assert all(sha(R / n) == h for n, h in {**plan['sources'], **plan['data']}.items())
    assert all(sha(ROOT / n) == h for n, h in plan['dependencies'].items())
    assert all((R / n).read_text() == text for n, text in sources())
    return plan


def build():
    assert not (R / 'PLAN.json').exists(); verify_parent()
    parent_seal = read(PARENT / 'SEALED.json')
    assert all(sha(PARENT / n) == h for n, h in parent_seal['files'].items())
    bs = batches(); worlds, paired = dev_worlds(); pools = read(R / 'palettes.json')
    assert all(len(set(p)) == 128 for p in pools.values()) and len(set().union(*map(set, pools.values()))) == 384
    assert pools['train'] == read(R.parent / '20261116_relation_solution/entity_palettes.json')['train']
    for step, index in enumerate(schedule()):
        for seeds in views_for(bs[index], step):
            for row in bs[index]['pairs'][str(bs[index]['mixed_depth'])]:
                assert not check(row)
                names = vocabulary(row['observation'], pools['train'], seeds['names'])
                assert len(set(names.values())) == len(names)
    used = {r['world_id'] for p in DATA.glob('worlds_*.jsonl') for r in rows(p)}
    for folder in R.parent.iterdir():
        p = folder / 'dev_worlds.json'
        if folder != R and p.is_file():
            used |= {w['world_id'] for ws in read(p).values() for w in ws}
    assert not used & {w['world_id'] for ws in worlds.values() for w in ws}
    datasets = {'batches.jsonl': bs}; obs, labels = cases()
    datasets.update({'observations.jsonl': obs, 'labels.jsonl': labels})
    third, third_labels = third_cases(); dev, dev_labels = dev_cases(worlds)
    assert len(third) == 168 and len(dev) == 1536 and sum(map(len, worlds.values())) == 128
    datasets.update({'third_observations.jsonl': third, 'third_labels.jsonl': third_labels,
                     'dev_observations.jsonl': dev, 'dev_labels.jsonl': dev_labels})
    for name, values in datasets.items():
        (R / name).write_text(''.join(json.dumps(v) + '\n' for v in values), encoding='utf-8')
    (R / 'dev_worlds.json').write_text(json.dumps(worlds), encoding='utf-8')
    deps = read(PARENT / 'PLAN.json')['dependencies'].copy()
    for folder in [PARENT, R.parent / '20261117_bit_weight_repair', R.parent / '20261117_query_coverage']:
        for p in folder.glob('*.py'): deps[p.relative_to(ROOT).as_posix()] = sha(p)
    for name in ['PLAN.json', 'SEALED.json', 'RESULTS.json']:
        deps[(PARENT / name).relative_to(ROOT).as_posix()] = sha(PARENT / name)
    for name in parent_seal['files']:
        if name.startswith('fit/joint/'):
            deps[(PARENT / name).relative_to(ROOT).as_posix()] = sha(PARENT / name)
    cp = PARENT / 'fit/joint'
    plan = {'arms': ['fixed', 'renamed'], 'evidence_arms': ['fixed', 'renamed'],
            'checkpoint': read(PARENT / 'PLAN.json')['checkpoint'],
            'fixed_checkpoint': {'path': cp.relative_to(ROOT).as_posix(), 'seal': read(cp / 'CHECKPOINT_SEALED.json')},
            'steps': 144, 'schedule': schedule(), 'lr': .0002, 'seed': 17, 'paired_conditions': paired,
            'source_sha256': sha(R / 'run.py'), 'parent_seal_sha256': sha(PARENT / 'SEALED.json'),
            'sources': {p.name: sha(p) for p in R.glob('*.py')},
            'data': {n: sha(R / n) for n in [*datasets, 'dev_worlds.json', 'palettes.json', 'PALETTE_RECEIPT.json', 'PLAN.md']},
            'dependencies': deps, 'new_fixed_panel_responses': 3888, 'reused_responses': 480,
            'episodes_per_arm': 128, 'whole_goal_complete': False, 'oracle_at_runtime': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'new_updates': 144, 'new_fixed_panel_responses': 3888, 'autonomous_episodes': 256, 'dependencies': len(deps)}))


if __name__ == '__main__':
    build()
