"""Finite one-round audit with all failure counts and grouped presentation metrics."""
import collections
import json
import re
from step_data import R, read, rows, sha, verify, cases, worlds


def parsed(row):
    matches = re.findall(r'ANSWER:\s*([01])\b', row['raw'])
    return matches[0] if len(matches) == 1 and row['seconds_batch'] < 9.5 else None


def score(observations, labels, raw):
    assert len(observations) == len(labels) == len(raw) == 320
    stats = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(list); lengths = collections.defaultdict(list)
    groups = collections.defaultdict(list); times = []; invalid = total_correct = 0
    for i, (o, label, answer) in enumerate(zip(observations, labels, raw)):
        assert o['id'] == label['id'] == answer['id']; bit = parsed(answer); correct = bit == str(label['target'])
        invalid += bit is None; total_correct += correct
        key = (o['kind'], o['noise'] if o['kind'] == 'switch' else None, o['vocab'])
        stats[key].update(queries=1, correct=int(correct), invalid=int(bit is None), positive=int(label['target'] == 1),
                          false_positive=int(bit == '1' and label['target'] == 0), false_negative=int(bit == '0' and label['target'] == 1))
        if o['kind'] == 'switch':
            pair_key = (o['group'], o['vocab'], o['view'])
            pairs[pair_key].append((o['side'], correct, key)); lengths[pair_key].append(answer['prompt_tokens'])
        if i % 4 == 0:
            assert len({r['seconds_batch'] for r in raw[i:i + 4]}) == 1
            assert all(r['padded_width'] == max(v['prompt_tokens'] for v in raw[i:i + 4]) for r in raw[i:i + 4])
            times.append(answer['seconds_batch'])
    assert len(pairs) == 128
    for (group, vocab, view), values in pairs.items():
        assert len(values) == 2 and {x[0] for x in values} == {0, 1} and values[0][2] == values[1][2]
        both = all(x[1] for x in values); stats[values[0][2]].update(pairs=1, both_correct=int(both)); groups[group].append(both)
    assert len(groups) == 32 and all(len(v) == 4 for v in groups.values())
    control_correct = sum(s['correct'] for (kind, _, _), s in stats.items() if kind != 'switch')
    token_control = all(len(set(v)) == 1 for v in lengths.values())
    gate = all(s['pairs'] == 16 and s['both_correct'] / s['pairs'] >= .95 for (kind, _, _), s in stats.items() if kind == 'switch')
    result = {'queries': len(raw), 'correct': total_correct, 'invalid': invalid, 'changing_pairs': 128,
              'paired_correct': sum(s['both_correct'] for (kind, _, _), s in stats.items() if kind == 'switch'),
              'all_four_groups': sum(all(v) for v in groups.values()), 'group_count': 32,
              'controls_correct': control_correct, 'controls': 64, 'pair_token_control': token_control,
              'max_prompt_tokens': max(r['prompt_tokens'] for r in raw),
              'local_gate_passed': gate and control_correct == 64 and invalid == 0 and token_control,
              'cells': [{'kind': k, 'noise': n, 'vocab': v, **dict(s)} for (k, n, v), s in stats.items()]}
    return result, times


def main():
    plan = verify(); seal = read(R / 'PROBE_SEALED.json'); obs, labels = cases()
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(R / n) == h for n, h in seal['files'].items())
    assert obs == rows(R / 'observations.jsonl') and labels == rows(R / 'labels.jsonl')
    train, dev, controls = worlds(); assert read(R / 'worlds.json') == {'train_pairs': train, 'dev_pairs': dev, 'controls': controls}
    results = []; times = []
    for arm in plan['arms']:
        identity = read(R / 'probe' / arm / 'IDENTITY.json'); assert identity['before'] == identity['after'] and identity['errors'] == 0
        for mode in ['zero_shot', 'four_examples']:
            selected = [(o, l) for o, l in zip(obs, labels) if o['mode'] == mode]
            result, elapsed = score([x[0] for x in selected], [x[1] for x in selected], rows(R / 'probe' / arm / (mode + '.jsonl')))
            results.append({'arm': arm, 'mode': mode, **result}); times.extend(elapsed)
    result = {'summaries': results, 'new_responses': 1280, 'new_updates': 0, 'generation_seconds': sum(times),
              'max_generation_seconds': max(times), 'errors': 0, 'autonomous_episodes': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'PROBE_RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({**result, 'summaries': [{k: v for k, v in s.items() if k != 'cells'} for s in results]}))


if __name__ == '__main__': main()
