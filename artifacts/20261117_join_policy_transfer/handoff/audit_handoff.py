import collections
import json
import math
import re
from join_handoff_data import H, R, read, rows, sha, verify


def main():
    plan = verify(); seal = read(H / 'SEALED.json'); token = read(H / 'TOKEN_PREFLIGHT.json')
    assert seal['complete'] and seal['plan_sha256'] == token['plan_sha256'] == sha(H / 'PLAN.json')
    assert seal['parent_seal_sha256'] == sha(R / 'SEALED.json') and token['script_sha256'] == sha(H / 'token_preflight.py')
    assert all(sha(H / n) == h for n, h in seal['files'].items())
    obs, labels = rows(H / 'observations.jsonl'), rows(H / 'labels.jsonl'); cps = read(R / 'study/PLAN.json')['checkpoint_bindings']; summaries = []; times = []
    for arm in plan['arms']:
        identity = read(H / 'outputs' / (arm + '_IDENTITY.json'))
        assert identity['before'] == identity['after'] == read(R / 'study' / arm / 'EVAL_IDENTITY.json')['before'] and identity['errors'] == 0
        assert identity['checkpoint'] == cps[arm]; raw = rows(H / 'outputs' / (arm + '.jsonl')); assert len(raw) == 32
        assert [v['prompt_tokens'] for v in raw] == token['lengths']; counts = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(dict)
        for i, (o, label, v) in enumerate(zip(obs, labels, raw)):
            assert o['id'] == label['id'] == v['id'] and math.isfinite(v['seconds_batch']) and v['seconds_batch'] > 0
            matches = re.findall(r'ANSWER:\s*([ABC])\b', v['raw']); choice = None
            if len(matches) == 1 and ord(matches[0]) - 65 < len(o['actions']) and v['seconds_batch'] < 9.5:
                choice = o['actions'][ord(matches[0]) - 65]
            counts[o['treatment']].update(queries=1, invalid=int(choice is None), true_optimal=int(choice == label['true_optimal_action']), follows_supplied_positive=int(choice == label['supplied_positive_action']))
            pairs[o['state_id']][o['treatment']] = choice
            if i % 4 == 0:
                batch = raw[i:i + 4]; assert len({x['seconds_batch'] for x in batch}) == 1
                assert all(x['padded_width'] == max(y['prompt_tokens'] for y in batch) for x in batch); times.append(v['seconds_batch'])
        assert len(pairs) == 16 and all(set(p) == {'truth', 'inverted'} for p in pairs.values())
        assert all(c['queries'] == 16 for c in counts.values())
        flips = sum(p['truth'] is not None and p['inverted'] is not None and p['truth'] != p['inverted'] for p in pairs.values())
        summaries.append({'arm': arm, 'scores': {k: dict(v) for k, v in counts.items()}, 'valid_action_flips': flips, 'states': 16})
    result = {'summaries': summaries, 'new_responses': 288, 'new_updates': 0, 'new_checkpoints': 0, 'reused_responses': 0, 'autonomous_episodes': 0,
              'generation_seconds': sum(times), 'max_generation_seconds': max(times), 'errors': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO',
              'scope': 'Injected judgment-channel intervention only; neither treatment contains model-generated judgments. Does not prove relational origin or faithful internal reasoning.'}
    (H / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__': main()
