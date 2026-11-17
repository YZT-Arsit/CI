import json
from factor_data import R, ROOT, read, rows, sha, verify, dataset
from scoring_factors import score


def main():
    plan = verify(); seal = read(R / 'SEALED.json'); token = read(R / 'TOKEN_PREFLIGHT.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json') and all(sha(R / n) == h for n, h in seal['files'].items())
    assert token['plan_sha256'] == sha(R / 'PLAN.json') and token['script_sha256'] == sha(R / 'token_preflight.py')
    data = dataset(); assert all(data[n] == rows(R / n) for n in data)
    summaries = []; times = []; count = 0
    for arm, cp in plan['models'].items():
        identity = read(R / 'outputs' / f'{arm}_IDENTITY.json')
        assert identity['before'] == identity['after'] == read(ROOT / cp['path'] / 'EVAL_IDENTITY.json')['before']
        assert identity['errors'] == 0 and identity['checkpoint'] == cp
        for mode in cp['modes']:
            prefix = 'echo_' if mode == 'echo' else ''; obs, labels = data[prefix + 'observations.jsonl'], data[prefix + 'labels.jsonl']
            raw = rows(R / 'outputs' / f'{arm}_{mode}.jsonl')
            result, elapsed = score(obs, labels, raw, mode)
            assert [r['prompt_tokens'] for r in raw] == token['lengths'][mode]
            summaries.append({'arm': arm, 'mode': mode, **result}); times.extend(elapsed); count += len(raw)
    assert count == plan['new_responses'] == 2688
    result = {'summaries': summaries, 'new_responses': count, 'new_updates': 0, 'new_checkpoints': 0, 'reused_responses': 0,
              'generation_seconds': sum(times), 'max_generation_seconds': max(times), 'errors': 0,
              'autonomous_episodes': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({**result, 'summaries': [{k: v for k, v in s.items() if k != 'cells'} for s in summaries]}))


if __name__ == '__main__': main()
