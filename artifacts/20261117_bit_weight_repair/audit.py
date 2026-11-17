"""Rescore the matched control and bit-weight repair without model inference."""
import collections
import itertools
import json
import math
import re
from repair_data import R, ROOT, PARENT, read, rows, sha, verify, batches, schedule, cases


def parse(answer, letters):
    found = re.findall(r'ANSWER:\s*([' + letters + r'])\b', answer['raw'])
    return found[0] if len(found) == 1 and answer['seconds_batch'] < 9.5 else None


def score(folder):
    observations, labels = cases(); raw = rows(folder / 'query_raw.jsonl')
    assert len(raw) == 336
    stats = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(list); lengths = collections.defaultdict(list); times = []
    for i, (o, label, answer) in enumerate(zip(observations, labels, raw)):
        assert o['id'] == label['id'] == answer['id']
        bit = parse(answer, '01'); correct = bit == str(label['target'])
        key = (o['vocab'], label['role'], label['depth'])
        stats[key].update(queries=1, correct=int(correct), invalid=int(bit is None))
        prefix = 'changing' if label['changing'] else 'fixed'
        stats[key].update({prefix + '_queries': 1, prefix + '_correct': int(correct), prefix + '_invalid': int(bit is None)})
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
    base = R.parent / '20261117_contrast_repair'
    obs = rows(base / 'fresh_observations.jsonl'); labels = rows(base / 'fresh_labels.jsonl'); raw = rows(folder / 'oracle_raw.jsonl')
    assert len(raw) == 144
    groups = collections.defaultdict(list); unsafe = invalid = 0
    for i, (o, label, answer) in enumerate(zip(obs, labels, raw)):
        assert o['id'] == label['id'] == answer['id']
        letter = parse(answer, 'ABC'); chosen = o['actions'][ord(letter) - 65] if letter else None
        case = label['case']; best = {case['candidates'][j]['action'] for j in case['optimal_indices']}
        groups[o['case_id']].append((o['group'], tuple(o['actions']), chosen in best))
        invalid += int(letter is None)
        unsafe += int(any(c['action'] == chosen and not c['immediate_safe'] for c in case['candidates']))
        if i % 4 == 0:
            assert len({x['seconds_batch'] for x in raw[i:i + 4]}) == 1
            times.append(answer['seconds_batch'])
    assert all(len(v) == 6 and {x[1] for x in v} == set(itertools.permutations(v[0][1])) for v in groups.values())
    oracle = {name: sum(all(x[2] for x in v) for v in groups.values() if v[0][0] == name) for name in ['zero', 'positive']}
    token_pass = all(len(set(x)) == 1 for x in lengths.values())
    return {'paired_correct': count, 'pairs_per_vocabulary': 48,
            'query_summaries': [{'vocab': v, 'role': role, 'depth': depth, **dict(s)} for (v, role, depth), s in stats.items()],
            'rename_token_control': token_pass, 'oracle_all_six': oracle, 'oracle_unsafe': unsafe, 'oracle_invalid': invalid,
            'original_fit_gate': count['original'] == 48, 'rename_gate': count['renamed'] == 48 and token_pass,
            'oracle_retention_gate': oracle == {'zero': 12, 'positive': 12} and unsafe == invalid == 0}, times


def main():
    plan = verify(); seal = read(R / 'SEALED.json'); folder = R / 'fit'
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(R / n) == h for n, h in seal['files'].items())
    assert plan['schedule'] == schedule() == read(PARENT / 'PLAN.json')['schedule']
    cp = read(folder / 'CHECKPOINT_SEALED.json')
    assert cp['steps'] == 144 and cp['initial_adapter_hashes'] == plan['checkpoint']['seal']['adapter_hashes']
    assert all(sha(folder / 'adapter' / n) == h for n, h in cp['adapter_hashes'].items())
    assert read(folder / 'INITIAL_STATE.json') == read(PARENT / 'fit/joint/INITIAL_STATE.json')
    identity = read(folder / 'EVAL_IDENTITY.json'); assert identity['before'] == identity['after'] and identity['errors'] == 0
    updates = rows(folder / 'training.jsonl'); original = rows(PARENT / 'fit/joint/training.jsonl'); bs = batches()
    assert len(updates) == len(original) == 144 and not (folder / 'TIMEOUT.json').exists()
    assert [x['tokens'] for x in updates] == [x['tokens'] for x in original]
    for i, (record, index) in enumerate(zip(updates, plan['schedule'])):
        assert record['step'] == record['optimizer_step'] == i + 1 and record['batch_index'] == index
        assert 0 < record['seconds'] < 15 and all(math.isfinite(record[k]) for k in ['loss', 'query_loss', 'actor_loss', 'gradient_norm'])
        assert abs(record['loss'] - record['query_loss'] - record['actor_loss']) < 1e-5
        batch = bs[index]; pair = batch['pairs'][str(batch['mixed_depth'])]
        teacher = [{c['action']: str(c['safe_completion']) for c in row['candidates']} for row in pair]
        assert record['supplied'] == [teacher, teacher] == original[i]['supplied']
    receipt = read(folder / 'CONTROL_RECEIPT.json')
    assert receipt == {'source': (PARENT / 'fit/joint').relative_to(ROOT).as_posix(), 'source_seal_sha256': sha(PARENT / 'SEALED.json'),
                       'source_results_sha256': sha(PARENT / 'RESULTS.json'), 'new_control_calls': 0}
    control, _ = score(PARENT / 'fit/joint'); repaired, times = score(folder)
    parent = next(s for s in read(PARENT / 'RESULTS.json')['summaries'] if s['arm'] == 'joint')
    assert control == {k: v for k, v in parent.items() if k != 'arm'}
    result = {'control': control, 'repair': repaired, 'new_updates': 144, 'new_responses': 480, 'reused_control_responses': 480,
              'training_seconds': sum(x['seconds'] for x in updates), 'max_update_seconds': max(x['seconds'] for x in updates),
              'generation_seconds': sum(times), 'max_generation_seconds': max(times), 'errors': 0,
              'full_autonomous_rollouts': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: ({a: b for a, b in v.items() if a != 'query_summaries'} if k in ['control', 'repair'] else v) for k, v in result.items()}))


if __name__ == '__main__':
    main()
