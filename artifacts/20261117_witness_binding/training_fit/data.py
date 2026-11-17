"""Select one complete fixed epoch of actual TRAIN presentations."""
import json
import sys
from pathlib import Path
F = Path(__file__).resolve().parent
sys.path.insert(0, str(F.parent))
from graph_data import R, ROOT, NAMES, read, rows, sha, verify, render


def build():
    verify(); assert not (F / 'PLAN.json').exists(); pool = read(NAMES / 'palettes.json')['train']
    all_obs = []; all_labels = []
    for seed in [17, 29, 41]:
        worlds = read(R / f'worlds_{seed}.json'); schedule = rows(R / f'schedule_{seed}.jsonl')[:12]
        assert len({tuple(b['stratum']) for b in schedule}) == 12
        for batch in schedule:
            for i, (key, name, target) in enumerate(zip(batch['ids'], batch['names'], batch['labels'])):
                w = worlds[key]; prompt, valid = render(w, pool, name); proof, valid2 = render(w, pool, name, 'proof'); assert valid == valid2
                identifier = f"train_{seed}_{batch['step']}_{i}"
                all_obs.append({'id': identifier, 'seed': seed, 'step': batch['step'], 'world_id': key, 'name_seed': name,
                                'kind': w['kind'], 'family': w['family'], 'noise': w['noise'], 'group': w['group'], 'side': w['side'],
                                'prompt': prompt, 'proof_prompt': proof})
                all_labels.append({'id': identifier, 'target': target, 'valid_witnesses': valid})
    assert len(all_obs) == 144 and sum(l['target'] for l in all_labels) == 72
    for name, data in [('observations.jsonl', all_obs), ('labels.jsonl', all_labels)]:
        (F / name).write_text(''.join(json.dumps(r) + '\n' for r in data), encoding='utf-8')
    plan = {'main_plan_sha256': sha(R / 'PLAN.json'), 'new_responses': 864, 'new_updates': 0, 'autonomous_episodes': 0,
            'data_hashes': {n: sha(F / n) for n in ['PLAN.md', 'observations.jsonl', 'labels.jsonl']},
            'source_hashes': {p.name: sha(p) for p in F.glob('*.py')}}
    (F / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'train_presentations': 144, 'planned_new_responses': 864}))


def verified():
    main = verify(); plan = read(F / 'PLAN.json')
    assert plan['main_plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(F / n) == h for n, h in {**plan['data_hashes'], **plan['source_hashes']}.items())
    return main, plan


if __name__ == '__main__': build()
