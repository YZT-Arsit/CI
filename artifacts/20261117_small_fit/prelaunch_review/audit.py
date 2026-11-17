"""Native rescore of every fit diagnostic response, with frozen-source verification."""
import collections
import itertools
import json
import math
import re
from fit_data import R, ROOT, PARENT, read, rows, sha, verify, batches, schedule, cases


def parsed(raw, letters):
    found = re.findall(r'ANSWER:\s*([' + letters + r'])\b', raw['raw'])
    return found[0] if len(found) == 1 and raw['seconds_batch'] < 9.5 else None


def main():
    plan = verify(); seal = read(R / 'SEALED.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(R / name) == value for name, value in seal['files'].items())
    bs = batches(); obs, labels = cases()
    assert bs == rows(R / 'batches.jsonl') and plan['schedule'] == schedule()
    assert obs == rows(R / 'observations.jsonl') and labels == rows(R / 'labels.jsonl')
    assert all(collections.Counter(plan['schedule'][i:i + 18]) == collections.Counter(range(18)) for i in range(0, 144, 18))
    original_steps = {b['step']: b for b in rows(PARENT / 'train_batches.jsonl')}
    assert all(original_steps[b['step']] == b for b in bs)
    summaries, times, training_times, initial_digests, token_sequences = [], [], [], [], []
    new_responses = 0
    for arm in plan['arms']:
        folder = R / 'fit' / arm
        initial_digests.append(read(folder / 'INITIAL_STATE.json')['trainable_sha256'])
        cp = read(folder / 'CHECKPOINT_SEALED.json')
        assert all(sha(folder / 'adapter' / name) == value for name, value in cp['adapter_hashes'].items())
        if arm == 'initial':
            assert cp == plan['checkpoint']['seal']
        else:
            assert cp['steps'] == 144 and cp['initial_adapter_hashes'] == plan['checkpoint']['seal']['adapter_hashes']
            updates = rows(folder / 'training.jsonl'); assert len(updates) == 144 and not (folder / 'TIMEOUT.json').exists()
            for i, (record, index) in enumerate(zip(updates, plan['schedule'])):
                assert record['step'] == record['optimizer_step'] == i + 1 and record['batch_index'] == index
                assert 0 < record['seconds'] < 15 and all(math.isfinite(record[k]) for k in ['loss', 'query_loss', 'actor_loss', 'gradient_norm'])
                expected = record['query_loss'] + (record['actor_loss'] if arm == 'joint' else 0.)
                assert abs(record['loss'] - expected) < 1e-5
                batch = bs[index]; pair = batch['pairs'][str(batch['mixed_depth'])]
                teacher = [{c['action']: str(c['safe_completion']) for c in row['candidates']} for row in pair]
                assert record['supplied'] == [teacher, teacher]
                training_times.append(record['seconds'])
            token_sequences.append([x['tokens'] for x in updates])
        identity = read(folder / 'EVAL_IDENTITY.json')
        assert identity['errors'] == 0 and identity['before'] == identity['after']
        raw = rows(folder / 'query_raw.jsonl'); assert len(raw) == 336
        stats = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(list); lengths = collections.defaultdict(list)
        for i, (o, label, answer) in enumerate(zip(obs, labels, raw)):
            assert o['id'] == label['id'] == answer['id']
            bit = parsed(answer, '01'); correct = bit == str(label['target'])
            key = (o['vocab'], label['role'], label['depth'])
            stats[key].update(queries=1, correct=int(correct), invalid=int(bit is None))
            if label['changing']: pairs[key, o['group']].append((o['side'], label['target'], correct))
            lengths[o['group']].append(answer['prompt_tokens'])
            if i % 4 == 0:
                assert len({x['seconds_batch'] for x in raw[i:i + 4]}) == 1
                times.append(answer['seconds_batch'])
        for (key, group), values in pairs.items():
            assert len(values) == 2 and {x[0] for x in values} == {0, 1} and {x[1] for x in values} == {0, 1}
            stats[key].update(pairs=1, both_correct=int(all(x[2] for x in values)))
        assert len(lengths) == 84 and all(len(x) == 4 for x in lengths.values())
        count = {v: sum(s['both_correct'] for (vocab, _, _), s in stats.items() if vocab == v) for v in ['original', 'renamed']}
        assert all(sum(s['pairs'] for (vocab, _, _), s in stats.items() if vocab == v) == 48 for v in count)
        new_responses += 336
        oracle_obs = rows(PARENT / 'fresh_observations.jsonl'); oracle_labels = rows(PARENT / 'fresh_labels.jsonl')
        oracle_raw = rows(folder / 'oracle_raw.jsonl'); assert len(oracle_raw) == 144
        groups = collections.defaultdict(list); unsafe = invalid = 0
        for i, (o, label, answer) in enumerate(zip(oracle_obs, oracle_labels, oracle_raw)):
            assert o['id'] == label['id'] == answer['id']
            letter = parsed(answer, 'ABC'); chosen = o['actions'][ord(letter) - 65] if letter else None
            case = label['case']; best = {case['candidates'][j]['action'] for j in case['optimal_indices']}
            groups[o['case_id']].append((o['group'], tuple(o['actions']), chosen in best))
            invalid += int(letter is None)
            unsafe += int(any(c['action'] == chosen and not c['immediate_safe'] for c in case['candidates']))
            if i % 4 == 0 and arm != 'initial':
                assert len({x['seconds_batch'] for x in oracle_raw[i:i + 4]}) == 1
                times.append(answer['seconds_batch'])
        assert all(len(v) == 6 and {x[1] for x in v} == set(itertools.permutations(v[0][1])) for v in groups.values())
        oracle = {name: sum(all(x[2] for x in v) for v in groups.values() if v[0][0] == name) for name in ['zero', 'positive']}
        if arm == 'initial':
            source = PARENT / 'repair/mixed_depth/fresh_raw.jsonl'
            assert (folder / 'oracle_raw.jsonl').read_bytes() == source.read_bytes()
            assert read(folder / 'REUSED_ORACLE.json') == {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source), 'new_calls': 0}
        else:
            new_responses += 144
        token_pass = all(len(set(x)) == 1 for x in lengths.values())
        summaries.append({'arm': arm, 'paired_correct': count, 'pairs_per_vocabulary': 48,
                          'query_summaries': [{'vocab': v, 'role': role, 'depth': depth, **dict(s)} for (v, role, depth), s in stats.items()],
                          'rename_token_control': token_pass, 'oracle_all_six': oracle, 'oracle_unsafe': unsafe, 'oracle_invalid': invalid,
                          'original_fit_gate': count['original'] == 48,
                          'rename_gate': count['renamed'] == 48 and token_pass,
                          'oracle_retention_gate': oracle == {'zero': 12, 'positive': 12} and unsafe == invalid == 0})
    assert len(set(initial_digests)) == 1 and token_sequences[0] == token_sequences[1]
    assert new_responses == 1296 and len(training_times) == 288
    result = {'summaries': summaries, 'new_responses': new_responses, 'reused_responses': 144,
              'new_updates': len(training_times), 'training_seconds': sum(training_times), 'max_update_seconds': max(training_times),
              'generation_seconds': sum(times), 'max_generation_seconds': max(times), 'errors': 0,
              'full_autonomous_rollouts': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({**result, 'summaries': [{k: v for k, v in s.items() if k != 'query_summaries'} for s in summaries]}))


if __name__ == '__main__':
    main()
