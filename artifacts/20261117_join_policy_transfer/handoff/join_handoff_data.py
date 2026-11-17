"""Inject truthful versus inverted judgments while holding initial facts fixed."""
import json
import sys
from pathlib import Path
H = Path(__file__).resolve().parent; R = H.parent
sys.path.insert(0, str(R))
import policy_data as data
ROOT, read, rows, sha = data.ROOT, data.read, data.rows, data.sha
from kci.relation_language import prompt, seed_for


def build():
    obs = []; labels = []; pool = read(R / 'palettes.json')['new']; worlds = read(R / 'dev_worlds.json'); index = 0
    for condition, cases in worlds.items():
        if not condition.startswith(('base_', 'depth3_')): continue
        for world in cases:
            env = data.environment(world); state = env.initial(); view = env.observation(state)
            case = next(c for c in data.targets(world) if c['observation'] == view)
            text, actions = prompt(view, pool, seed_for(world['base_group_id'], 0)); assert len(actions) == 2
            true = {c['action']: int(c['safe_completion']) for c in case['candidates']}
            assert sorted(true.values()) == [0, 1] and len(case['optimal_indices']) == 1
            correct = case['candidates'][case['optimal_indices'][0]]['action']; assert true[correct] == 1
            treatments = ['truth', 'inverted'] if index % 2 == 0 else ['inverted', 'truth']; pair = []
            for treatment in treatments:
                bits = true if treatment == 'truth' else {a: 1 - v for a, v in true.items()}
                actor = text + 'Own estimates (1 means compliant completion remains possible): ' + '; '.join(chr(65 + i) + '=' + str(bits[a]) for i, a in enumerate(actions)) + '.\n\nReply ANSWER:\nletter.'
                ident = condition + '/' + world['world_id'] + '/' + treatment
                obs.append({'id': ident, 'state_id': condition + '/' + world['world_id'], 'condition': condition, 'world_id': world['world_id'],
                            'treatment': treatment, 'prompt': actor, 'actions': actions})
                labels.append({'id': ident, 'true_optimal_action': correct, 'supplied_positive_action': next(a for a in actions if bits[a] == 1)})
                pair.append(actor)
            assert len(pair[0]) == len(pair[1]) and sum(a != b for a, b in zip(*pair)) == 2
            index += 1
    assert index == 16 and len(obs) == len(labels) == 32 and len({o['id'] for o in obs}) == 32
    return obs, labels


def verify():
    parent = data.verify(); plan = read(H / 'PLAN.json'); seal = read(R / 'SEALED.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json') and plan['parent_seal_sha256'] == sha(R / 'SEALED.json')
    assert all(sha(R / n) == h for n, h in seal['files'].items())
    assert plan['arms'] == parent['arms'] and plan['parent_plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(H / n) == h for n, h in {**plan['sources'], **plan['data']}.items())
    obs, labels = build(); assert obs == rows(H / 'observations.jsonl') and labels == rows(H / 'labels.jsonl')
    return plan


if __name__ == '__main__':
    assert not (H / 'PLAN.json').exists(); parent = data.verify(); seal = read(R / 'SEALED.json'); assert seal['complete']
    assert seal['plan_sha256'] == sha(R / 'PLAN.json') and all(sha(R / n) == h for n, h in seal['files'].items())
    for name, values in zip(['observations.jsonl', 'labels.jsonl'], build()):
        (H / name).write_text(''.join(json.dumps(v) + '\n' for v in values), encoding='utf-8')
    plan = {'arms': parent['arms'], 'parent_plan_sha256': sha(R / 'PLAN.json'), 'parent_seal_sha256': sha(R / 'SEALED.json'),
            'new_responses': 288, 'new_updates': 0, 'new_checkpoints': 0, 'autonomous_episodes': 0,
            'sources': {n: sha(H / n) for n in ['join_handoff_data.py', 'run_handoff.py', 'audit_handoff.py', 'token_preflight.py', 'PLAN.md']},
            'data': {n: sha(H / n) for n in ['observations.jsonl', 'labels.jsonl']}}
    (H / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({'states': 16, 'prompts': 32, 'new_responses': 288, 'plan_sha256': sha(H / 'PLAN.json')}))
