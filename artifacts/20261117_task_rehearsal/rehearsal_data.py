"""Matched original-task rehearsal versus extra local questions; developed DEV."""
import importlib.util
import json
import random
import sys
from pathlib import Path

R = Path(__file__).resolve().parent; ROOT = R.parents[1]; PARENT = R.parent / '20261117_join_repair'
sys.path.insert(0, str(PARENT))
import join_data
read, rows, sha, prior = join_data.read, join_data.rows, join_data.sha, join_data.prior
spec = importlib.util.spec_from_file_location('previous_native_panel', R.parent / '20261117_join_policy_transfer/policy_data.py')
policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
targets, environment, check = policy.targets, policy.environment, policy.check
from kci.relation_language import prompt


def training(seed):
    bs = rows(R.parent / '20261117_small_fit/batches.jsonl')
    assert len(bs) == 18
    local_worlds = read(PARENT / f'worlds_{seed}.json')['targeted']
    old = rows(PARENT / f'schedule_{seed}.jsonl')[:54]
    pool = read(prior.NAMES / 'palettes.json')['train']; rng = random.Random(690000000 + seed)
    order = []
    for cycle in range(3):
        indices = list(range(18)); rng.shuffle(indices); order += indices
    result = []
    for step, (batch, index) in enumerate(zip(old, order)):
        item = {'step': step, 'common': [], 'local': [], 'global': [], 'actor': batch['actor']}
        for key in ['common', 'local']:
            # Each original paired side shares a name seed. Controls may also share it.
            names = [n for _ in range(2) for n in [rng.randrange(10**9)] * 2]
            for identifier, name, label in zip(batch['ids'], names, batch['labels']):
                item[key].append({'world_id': identifier, 'name_seed': name, 'label': label,
                                  'prompt': prior.render(local_worlds[identifier], pool, name)[0]})
        b = bs[index]; pair = b['pairs'][str(b['mixed_depth'])]
        changing = [i for i, (a, c) in enumerate(zip(pair[0]['candidates'], pair[1]['candidates'])) if a['safe_completion'] != c['safe_completion']]
        assert changing
        item['global_cell'] = [b['key'][0], b['mixed_depth'], b['factors'][0]]
        item['global_batch_index'] = index
        for view in range(2):
            ci = changing[view % len(changing)]; name = rng.randrange(10**9)
            for side, case in enumerate(pair):
                c = case['candidates'][ci]
                text, _ = prompt(case['observation'], pool, name, question='task', candidate=c['action'])
                text = text.replace('A yes; B no.', 'Reply ANSWER: followed by 1 if possible, otherwise 0.')
                item['global'].append({'case': case, 'candidate_index': ci, 'name_seed': name,
                                       'view': view, 'side': side, 'label': c['safe_completion'], 'prompt': text})
        result.append(item)
    return result


def worlds():
    # Same developed condition design, new source IDs/name nonces; not new semantics.
    paired = ['base_v0', 'base_v1', 'depth3_v0', 'depth3_v1']
    import copy
    result = {}
    for family, depth in [('base', 1), ('depth3', 3)]:
        for view in [0, 1]:
            key = family + '_v' + str(view); result[key] = []; result[key + '_swap'] = []
        for n, task in enumerate(['meeting', 'query']):
            a = policy.make_world(680000000 + (1000 if depth == 3 else 0) + n, task, depth=depth, noise=2, mode='one', allow=False, required=True, delivered=True)
            a['base_group_id'] = a['world_id']
            for view in [0, 1]:
                for side, world in enumerate([a, policy.intervene(a, 'swap')]):
                    w = copy.deepcopy(world); w['base_group_id'] += '_view' + str(view)
                    result[family + '_v' + str(view) + ('_swap' if side else '')].append(w)
    for index, condition in enumerate(['optional', 'authorized', 'draft', 'both']):
        for view in [0, 1]: result[condition + '_v' + str(view)] = []
        for n, task in enumerate(['meeting', 'query']):
            w = policy.make_world(680010000 + index * 1000 + n, task, depth=1, noise=2, mode='both' if condition == 'both' else 'one',
                                  allow=condition == 'authorized', required=condition != 'optional', delivered=condition != 'draft')
            for view in [0, 1]:
                v = copy.deepcopy(w); v['base_group_id'] = v['world_id'] + '_view' + str(view)
                result[condition + '_v' + str(view)].append(v)
    return result, paired


def verify():
    join_data.verify(); plan = read(R / 'PLAN.json')
    assert sha(PARENT / 'SEALED.json') == plan['parent_seal_sha256']
    parent = read(PARENT / 'SEALED.json')
    assert parent['complete'] and all(sha(PARENT / n) == h for n, h in parent['files'].items())
    assert all(sha(ROOT / n) == h for n, h in plan['dependencies'].items())
    assert all(sha(R / n) == h for n, h in {**plan['sources'], **plan['data']}.items())
    return plan


def build():
    assert not (R / 'PLAN.json').exists(); join_data.verify()
    parent = read(PARENT / 'SEALED.json')
    assert parent['complete'] and all(sha(PARENT / n) == h for n, h in parent['files'].items())
    for seed in [17, 29, 41]:
        (R / f'training_{seed}.jsonl').write_text(''.join(json.dumps(b) + '\n' for b in training(seed)), encoding='utf-8')
    ws, paired = worlds(); (R / 'dev_worlds.json').write_text(json.dumps(ws, indent=2), encoding='utf-8')
    (R / 'palettes.json').write_bytes((prior.NAMES / 'palettes.json').read_bytes())
    checkpoints = {}
    for seed in [17, 29, 41]:
        cp = PARENT / 'study' / f'targeted{seed}'
        checkpoints[str(seed)] = {'path': cp.relative_to(ROOT).as_posix(), 'seal': read(cp / 'CHECKPOINT_SEALED.json')}
    deps = [PARENT / 'run_join.py', PARENT / 'join_data.py', R.parent / '20261117_small_fit/batches.jsonl',
            R.parent / '20261117_join_policy_transfer/policy_data.py', PARENT / 'audit_join.py']
    deps += list((R.parent / '20261117_join_policy_transfer').glob('*.py'))
    deps += list((ROOT / 'code/kci').glob('*.py'))
    deps += [R.parent / '20261117_action_binding/learned_evidence.py', R.parent / '20261116_relation_solution/audit_data.py',
             ROOT / 'code/experiments/receipt_dataset.py', ROOT / 'code/experiments/relation_solution_data.py']
    plan = {'arms': {f'{kind}{seed}': {'seed': seed, 'kind': kind} for seed in [17, 29, 41] for kind in ['initial', 'local', 'global']},
            'checkpoints': checkpoints, 'steps': 54, 'lr': .0002, 'new_updates': 324, 'new_checkpoints': 6,
            'new_fixed_responses': 5472, 'reused_fixed_responses': 2736, 'autonomous_episodes': 288,
            'paired_conditions': paired, 'parent_seal_sha256': sha(PARENT / 'SEALED.json'),
            'dependencies': {p.relative_to(ROOT).as_posix(): sha(p) for p in deps},
            'sources': {p.name: sha(p) for p in R.glob('*.py')},
            'data': {p.name: sha(p) for p in R.iterdir() if p.suffix in ['.jsonl', '.json', '.md']},
            'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'model_calls_before_freeze': 0, 'plan_sha256': sha(R / 'PLAN.json')}))


if __name__ == '__main__':
    build()
