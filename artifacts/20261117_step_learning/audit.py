"""Local learnability and conditional actor retention, never full-policy success."""
import collections
import itertools
import importlib.util
import json
import math
import re
from learning_data import R, ROOT, PARENT, NAMES, read, rows, sha, verify, train_worlds, training_schedule, confirmation_cases, label
spec = importlib.util.spec_from_file_location('local_step_scoring', PARENT / 'audit_probe.py')
scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
score = scorer.score


def oracle_score(folder):
    base = R.parent / '20261117_contrast_repair'; obs = rows(base / 'fresh_observations.jsonl'); labels = rows(base / 'fresh_labels.jsonl')
    raw = rows(folder / 'oracle.jsonl'); assert len(raw) == len(obs) == len(labels) == 144
    groups = collections.defaultdict(list); invalid = unsafe = 0; times = []
    for i, (o, l, answer) in enumerate(zip(obs, labels, raw)):
        assert o['id'] == l['id'] == answer['id']; found = re.findall(r'ANSWER:\s*([ABC])\b', answer['raw'])
        valid = len(found) == 1 and answer['seconds_batch'] < 9.5
        chosen = o['actions'][ord(found[0]) - 65] if valid else None; case = l['case']
        best = {case['candidates'][j]['action'] for j in case['optimal_indices']}
        groups[o['case_id']].append((o['group'], tuple(o['actions']), chosen in best)); invalid += not valid
        unsafe += any(c['action'] == chosen and not c['immediate_safe'] for c in case['candidates'])
        if i % 4 == 0:
            assert len({r['seconds_batch'] for r in raw[i:i + 4]}) == 1; times.append(answer['seconds_batch'])
    assert all(len(v) == 6 and {r[1] for r in v} == set(itertools.permutations(v[0][1])) for v in groups.values())
    counts = {g: sum(all(r[2] for r in v) for v in groups.values() if v[0][0] == g) for g in ['zero', 'positive']}
    return {'all_six': counts, 'invalid': invalid, 'unsafe': unsafe, 'retained': counts == {'zero': 12, 'positive': 12} and invalid == unsafe == 0}, times


def main():
    plan = verify(); seal = read(R / 'SEALED.json'); folder = R / 'learning'
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json') and all(sha(R / n) == h for n, h in seal['files'].items())
    data = train_worlds(); obs, labels = confirmation_cases()
    assert data == read(R / 'training_worlds.json') and obs == rows(R / 'observations.jsonl') and labels == rows(R / 'labels.jsonl')
    summaries = []; times = []; training_times = []; initial_hashes = []; responses = 0
    for arm in plan['arms']:
        target = folder / arm; initial_hashes.append(read(target / 'INITIAL_STATE.json')['trainable_sha256'])
        cp = read(target / 'CHECKPOINT_SEALED.json'); assert all(sha(target / 'adapter' / n) == h for n, h in cp['adapter_hashes'].items())
        if arm == 'initial':
            assert cp == plan['checkpoint']['seal']; source = NAMES / 'study/renamed/oracle_raw.jsonl'
            assert (target / 'oracle.jsonl').read_bytes() == source.read_bytes()
            assert read(target / 'REUSED_ORACLE.json') == {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source), 'new_calls': 0}
        else:
            seed = int(arm.removeprefix('seed')); schedule = training_schedule(seed); updates = rows(target / 'training.jsonl')
            assert schedule == rows(R / f'schedule_{seed}.jsonl') and len(updates) == 144 and not (target / 'TIMEOUT.json').exists()
            assert cp['steps'] == 144 and cp['initial_adapter_hashes'] == plan['checkpoint']['seal']['adapter_hashes']
            for index, (record, batch) in enumerate(zip(updates, schedule)):
                assert all(record[k] == v for k, v in batch.items()) and record['step'] == index and record['optimizer_step'] == index + 1
                assert [label(data[key]) for key in record['ids']] == record['labels'] and collections.Counter(record['labels']) == {0: 2, 1: 2}
                assert 0 < record['seconds'] < 15 and all(math.isfinite(record[k]) for k in ['loss', 'bit_loss', 'format_loss', 'gradient_norm'])
                assert abs(record['loss'] - record['bit_loss'] - record['format_loss']) < 1e-5; training_times.append(record['seconds'])
        identity = read(target / 'EVAL_IDENTITY.json'); assert identity['before'] == identity['after'] and identity['errors'] == 0
        operator = []
        for mode in ['zero_shot', 'four_examples']:
            selected = [(o, l) for o, l in zip(obs, labels) if o['mode'] == mode]
            result, elapsed = score([x[0] for x in selected], [x[1] for x in selected], rows(target / (mode + '.jsonl')))
            operator.append({'mode': mode, **result}); times.extend(elapsed); responses += 320
        retained, elapsed = oracle_score(target)
        if arm != 'initial': times.extend(elapsed); responses += 144
        summaries.append({'arm': arm, 'operator': operator, 'oracle': retained,
                          'necessary_learning_gate_passed': any(x['local_gate_passed'] for x in operator) and retained['retained']})
    assert len(set(initial_hashes)) == 1 and len(training_times) == 432 and responses == 2992
    result = {'summaries': summaries, 'all_trained_passed': all(s['necessary_learning_gate_passed'] for s in summaries if s['arm'] != 'initial'),
              'new_updates': 432, 'new_responses': responses, 'reused_responses': 144,
              'training_seconds': sum(training_times), 'max_update_seconds': max(training_times),
              'generation_seconds': sum(times), 'max_generation_seconds': max(times), 'errors': 0,
              'autonomous_episodes': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({**result, 'summaries': [{**s, 'operator': [{k: v for k, v in x.items() if k != 'cells'} for x in s['operator']]} for s in summaries]}))


if __name__ == '__main__': main()
