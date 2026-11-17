import importlib.util
import json
import sys
from pathlib import Path
F = Path(__file__).resolve().parent; R = F.parent
sys.path.insert(0, str(R))
import join_data as data
ROOT, read, rows, sha = data.ROOT, data.read, data.rows, data.sha


def build():
    obs = []; labels = []; pool = read(data.prior.NAMES / 'palettes.json')['train']
    for seed in [17, 29, 41]:
        worlds = read(R / f'worlds_{seed}.json'); batches = rows(R / f'schedule_{seed}.jsonl')[:8]
        assert len({tuple(b['stratum']) for b in batches}) == 8
        for step, batch in enumerate(batches):
            kinds = ['random'] if batch['stratum'][0] == 'control' else ['random', 'targeted']
            for kind in kinds:
                for key, name, label in zip(batch['ids'], batch['names'], batch['labels']):
                    w = worlds[kind][key]; text = data.prior.render(w, pool, name)[0]
                    ident = f'{seed}/{step}/{kind}/{key}'
                    obs.append({'id': ident, 'seed': seed, 'step': step, 'background': 'shared' if len(kinds) == 1 else kind,
                                'world_id': key, 'name_seed': name, 'group': w['group'], 'side': w.get('side'),
                                'kind': w['kind'], 'prompt': text})
                    labels.append({'id': ident, 'target': label})
    assert len(obs) == len(labels) == 168 and len({o['id'] for o in obs}) == 168
    return obs, labels


def verified():
    parent = data.verify(); plan = read(F / 'PLAN.json')
    assert plan['parent_plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(F / n) == h for n, h in plan['sources'].items())
    assert all(sha(F / n) == h for n, h in plan['data'].items())
    obs, labels = build(); assert obs == rows(F / 'observations.jsonl') and labels == rows(F / 'labels.jsonl')
    assert plan['arms'] == {a: c for a, c in parent['arms'].items() if c['kind'] != 'initial'}
    return parent, plan


if __name__ == '__main__':
    assert not (F / 'PLAN.json').exists(); parent = data.verify()
    for name, values in zip(['observations.jsonl', 'labels.jsonl'], build()):
        (F / name).write_text(''.join(json.dumps(v) + '\n' for v in values), encoding='utf-8')
    plan = {'parent_plan_sha256': sha(R / 'PLAN.json'), 'arms': {a: c for a, c in parent['arms'].items() if c['kind'] != 'initial'},
            'selection': 'First eight batches, one full first epoch; two switch backgrounds and shared controls once, original TRAIN names/order.',
            'new_responses': 336, 'new_updates': 0, 'new_checkpoints': 0, 'whole_goal_complete': False,
            'sources': {n: sha(F / n) for n in ['join_fit_data.py', 'run_fit.py', 'audit_fit.py', 'PLAN.md']},
            'data': {n: sha(F / n) for n in ['observations.jsonl', 'labels.jsonl']}}
    (F / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'queries': 168, 'new_responses': 336, 'plan_sha256': sha(F / 'PLAN.json')}))
