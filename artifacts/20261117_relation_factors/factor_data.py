"""Matched local relation interventions; no runtime oracle replacement."""
import importlib.util
import json
import random
from pathlib import Path
R = Path(__file__).resolve().parent; ROOT = R.parents[1]; PARENT = R.parent / '20261117_witness_binding'
spec = importlib.util.spec_from_file_location('witness_data_for_factors', PARENT / 'graph_data.py')
prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
read, rows, sha = prior.read, prior.rows, prior.sha


def pair(seed, family, known, noise):
    assert known in [1, 4] and noise in ['none', 'irrelevant', 'near_miss']
    rng = random.Random(seed); roles = rng.sample(range(16), 16); a, b, target, other = roles[:4]; extra_known = roles[4:7]
    if family == 'retention': frontiers = [[target], [a]]; cores = [[], []]
    elif family == 'direction': frontiers = [[a], [a]]; cores = [[(a, target)], [(target, a)]]
    elif family == 'binding': frontiers = [[a], [b]]; cores = [[(a, target), (b, other)]] * 2
    else: raise ValueError(family)
    union = set(frontiers[0] + frontiers[1]); forbidden = set(cores[0] + cores[1]); edges = []
    if noise == 'irrelevant':
        pool = roles[7:]; edges = rng.sample([(u, v) for u in pool for v in pool if u != v], 12)
    elif noise == 'near_miss':
        # Fixed across frontier sizes; added known nodes cannot become witnesses.
        sources = sorted(union); destinations = [v for v in range(16) if v != target and v not in extra_known]
        outgoing = [(u, v) for u in sources for v in destinations if u != v and (u, v) not in forbidden]
        incoming = [(u, target) for u in range(16) if u not in union | set(extra_known) | {target} and (u, target) not in forbidden]
        edges = rng.sample(outgoing, 6) + rng.sample(incoming, 6)
    worlds = []
    for side in [0, 1]:
        frontier = frontiers[side] + (extra_known if known == 4 else [])
        if known == 4:
            core = frontier.pop(0); frontier.insert(seed % 4, core)
        worlds.append({'n': 16, 'frontier': frontier, 'edges': [list(e) for e in cores[side] + edges], 'target': target,
                       'seed': seed, 'family': family, 'known': known, 'noise': noise, 'side': side})
    assert [prior.truth(w) for w in worlds] == [1, 0]
    assert [len(prior.witnesses(w)) for w in worlds] == [1, 0]
    return worlds


def render(w, pool, mode):
    names = random.Random(w['seed'] + 17).sample(pool, 16)
    order = list(range(len(w['edges']))); random.Random(w['seed'] + 29).shuffle(order)
    fields = ('Known nodes: ' + ', '.join(names[v] for v in w['frontier']) + '.\nEdges: ' +
              '; '.join(names[w['edges'][i][0]] + ' -> ' + names[w['edges'][i][1]] for i in order) + '.\n')
    text = ('Directed edges go from the first name to the second. Start with the known nodes. '
            'Keep them and add destinations of edges whose source was known at the start. '
            'Do exactly one round; do not follow a second edge in this round.\n' + fields)
    valid = sorted(names[v] for v in prior.witnesses(w)) or ['none']
    if mode == 'binary':
        text += 'Is ' + names[w['target']] + ' known after this one round? Reply ANSWER: followed by 1 for yes or 0 for no.'
    elif mode == 'proof':
        text += ('Target: ' + names[w['target']] + '. Give one known node that proves the target is known after this round. '
                 'Valid choices are the target itself if already known, or any known source with an edge to the target. '
                 'If none exists, give none. Reply only PROOF: followed by that name or none.')
    else:
        assert mode == 'echo'; valid = [names[w['target']]]
        text = ('This is a copying check, not a graph inference question.\n' + fields + 'Target: ' + names[w['target']] +
                '. Copy the target name. Reply only PROOF: followed by that name.')
    return text, valid


def dataset():
    pools = read(prior.NAMES / 'palettes.json'); obs = []; labels = []; echo_obs = []; echo_labels = []
    for fi, family in enumerate(['retention', 'direction', 'binding']):
        for i in range(4):
            seed = 520000000 + fi * 1000 + i
            for known in [1, 4]:
                for noise in ['none', 'irrelevant', 'near_miss']:
                    for vocab, key in [('familiar', 'train'), ('third', 'new')]:
                        for w in pair(seed, family, known, noise):
                            identifier = f"{seed}_{known}_{noise}_{vocab}_{w['side']}"
                            binary, valid = render(w, pools[key], 'binary'); proof, _ = render(w, pools[key], 'proof')
                            meta = {'id': identifier, 'seed': seed, 'family': family, 'known': known, 'noise': noise, 'vocab': vocab, 'side': w['side']}
                            obs.append(meta | {'prompt': binary, 'proof_prompt': proof})
                            labels.append({'id': identifier, 'target': prior.truth(w), 'valid_witnesses': valid, 'world': w})
                            if family != 'binding' and known == 4 and noise == 'near_miss':
                                echo, answer = render(w, pools[key], 'echo')
                                echo_obs.append(meta | {'prompt': echo}); echo_labels.append({'id': identifier, 'valid_witnesses': answer})
    assert len(obs) == 288 and len(echo_obs) == 32
    return {'observations.jsonl': obs, 'labels.jsonl': labels, 'echo_observations.jsonl': echo_obs, 'echo_labels.jsonl': echo_labels}


def verify():
    prior.verify(); plan = read(R / 'PLAN.json'); parent = read(PARENT / 'SEALED.json')
    assert plan['parent_plan_sha256'] == sha(PARENT / 'PLAN.json') and plan['parent_seal_sha256'] == sha(PARENT / 'SEALED.json')
    assert parent['complete'] and all(sha(PARENT / n) == h for n, h in parent['files'].items())
    assert all(sha(R / n) == h for n, h in {**plan['sources'], **plan['data']}.items())
    return plan


def build():
    prior.verify(); assert not (R / 'PLAN.json').exists(); data = dataset()
    for name, values in data.items(): (R / name).write_text(''.join(json.dumps(v) + '\n' for v in values), encoding='utf-8')
    models = {}
    for seed in [17, 29, 41]:
        for kind in ['bit', 'proof']:
            arm = f'{kind}{seed}'; path = PARENT / 'study' / arm
            models[arm] = {'path': path.relative_to(ROOT).as_posix(), 'seal': read(path / 'CHECKPOINT_SEALED.json'),
                           'modes': ['binary'] if kind == 'bit' else ['binary', 'proof', 'echo']}
    plan = {'models': models, 'parent_plan_sha256': sha(PARENT / 'PLAN.json'), 'parent_seal_sha256': sha(PARENT / 'SEALED.json'),
            'sources': {p.name: sha(p) for p in R.glob('*.py')}, 'data': {n: sha(R / n) for n in [*data, 'PLAN.md']},
            'new_responses': 2688, 'new_updates': 0, 'whole_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'queries': 288, 'echo': 32, 'new_responses': 2688}))


if __name__ == '__main__': build()
