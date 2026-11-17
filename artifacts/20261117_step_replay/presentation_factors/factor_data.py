"""Independent name and order seeds with exact old-prompt reconstruction."""
import collections
import json
import random
import sys
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent))
from replay_data import ROOT, PARENT, NAMES, read, rows, sha, verify as verify_main, train_worlds, prompt as original_prompt, label


def render(world, pool, name_seed, order_seed):
    names = random.Random(name_seed).sample(pool, 12)
    rng = random.Random(order_seed); rng.sample(range(len(pool)), 12)
    frontier = world['frontier'][:]; edges = world['edges'][:]; rng.shuffle(frontier); rng.shuffle(edges)
    return ('Directed edges go from the first name to the second. Start with the known nodes. '
            'Keep them and add destinations of edges whose source was known at the start. '
            'Do exactly one round; do not follow a second edge in this round.\n'
            'Known nodes: ' + (', '.join(names[x] for x in frontier) or 'none') + '.\n'
            'Edges: ' + '; '.join(names[a] + ' -> ' + names[b] for a, b in edges) + '.\n'
            'Is ' + names[world['target']] + ' known after this one round? Reply ANSWER: followed by 1 for yes or 0 for no.'), names, frontier, edges


def cases():
    data = train_worlds(); pool = read(NAMES / 'palettes.json')['train']
    seen = [o for o in rows(PARENT / 'fit_probe/cases.jsonl') if o['view'] == 'seen']; assert len(seen) == 48
    obs = []; labels = []
    for seed in [17, 29, 41]:
        panel = [o for o in seen if o['seed'] == seed]
        for variant in ['seen', 'names_only', 'order_only', 'both']:
            for old in panel:
                world = data[old['world_id']]; source_seed = old['name_seed']
                text, original_names, _, original_edges = render(world, pool, source_seed, source_seed)
                assert text == old['prompt'] == original_prompt(world, pool, source_seed)
                selection_index = int(old['id'].split('_')[-2]); start = 490000000 + seed * 1000 + selection_index * 10
                name_seed = next(n for n in range(start, start + 1000) if all(a != b for a, b in zip(original_names, random.Random(n).sample(pool, 12))))
                # Select ordering using a pair's shared positive world, keeping both sides matched.
                anchor = next((data[o['world_id']] for o in panel if o['pair'] == old['pair'] and o['pair'] is not None), world)
                old_edges = render(anchor, pool, source_seed, source_seed)[3]
                order_seed = next(n for n in range(start, start + 1000) if render(anchor, pool, source_seed, n)[3] != old_edges)
                n = source_seed if variant in ['seen', 'order_only'] else name_seed
                o = source_seed if variant in ['seen', 'names_only'] else order_seed
                text, names, frontier, edges = render(world, pool, n, o)
                assert sorted(frontier) == sorted(world['frontier']) and sorted(edges) == sorted(world['edges'])
                assert (names == original_names) == (variant in ['seen', 'order_only'])
                if variant in ['seen', 'names_only']: assert edges == original_edges
                else: assert edges != original_edges
                identifier = f"{seed}_{variant}_{old['id'].split('_')[-2]}_{old['id'].split('_')[-1]}"
                obs.append({'id': identifier, 'seed': seed, 'view': variant, 'world_id': old['world_id'], 'kind': old['kind'], 'pair': old['pair'],
                            'name_seed': n, 'order_seed': o, 'prompt': text})
                labels.append({'id': identifier, 'target': label(world)})
    assert len(obs) == 192
    return obs, labels


def build():
    verify_main(); assert not (R / 'PLAN.json').exists(); obs, labels = cases()
    for name, values in [('observations.jsonl', obs), ('labels.jsonl', labels)]:
        (R / name).write_text(''.join(json.dumps(r) + '\n' for r in values), encoding='utf-8')
    plan = {'main_plan_sha256': sha(R.parent / 'PLAN.json'), 'source_fit_cases_sha256': sha(PARENT / 'fit_probe/cases.jsonl'),
            'files': {n: sha(R / n) for n in ['PLAN.md', 'factor_data.py', 'probe.py', 'observations.jsonl', 'labels.jsonl']},
            'arms': {'initial': None, 'local17': 17, 'local29': 29, 'local41': 41, 'replay17': 17, 'replay29': 29, 'replay41': 41},
            'new_responses': 576, 'new_updates': 0}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8'); print(json.dumps({'new_responses': 576, 'new_updates': 0}))


def check():
    verify_main(); plan = read(R / 'PLAN.json')
    assert plan['main_plan_sha256'] == sha(R.parent / 'PLAN.json') and plan['source_fit_cases_sha256'] == sha(PARENT / 'fit_probe/cases.jsonl')
    assert all(sha(R / n) == h for n, h in plan['files'].items())
    obs, labels = cases(); assert obs == rows(R / 'observations.jsonl') and labels == rows(R / 'labels.jsonl')
    return plan


if __name__ == '__main__': build()
