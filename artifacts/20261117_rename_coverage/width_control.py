"""Separate, post-registration batch-width calibration; never replace the main gate."""
import collections
import json
import sys
from rename_data import R, ROOT, read, rows, sha, verify, cases, third_cases
C = R / 'width_control'


def observations():
    third = {(o['group'], o['side']): o for o in third_cases()[0]}
    return [o if o['vocab'] == 'original' else third[o['group'], o['side']] for o in cases()[0]]


def verify_control():
    plan = verify(); registration = read(C / 'PLAN.json')
    assert sha(R / 'PLAN.json') == registration['main_plan_sha256'] and sha(R / 'SEALED.json') == registration['main_seal_sha256']
    assert sha(R / 'width_control.py') == registration['source_sha256'] and sha(R / 'WIDTH_CONTROL_PLAN.md') == registration['protocol_sha256']
    assert sha(C / 'observations.jsonl') == registration['observations_sha256'] and rows(C / 'observations.jsonl') == observations()
    return plan, registration


def build():
    plan = verify(); seal = read(R / 'SEALED.json')
    assert seal['complete'] and all(sha(R / n) == h for n, h in seal['files'].items())
    assert not C.exists(); C.mkdir(); obs = observations(); assert len(obs) == 336
    (C / 'observations.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in obs), encoding='utf-8')
    registration = {'arms': plan['arms'], 'new_responses': 672, 'new_updates': 0,
                    'main_plan_sha256': sha(R / 'PLAN.json'), 'main_seal_sha256': sha(R / 'SEALED.json'),
                    'source_sha256': sha(R / 'width_control.py'), 'protocol_sha256': sha(R / 'WIDTH_CONTROL_PLAN.md'),
                    'observations_sha256': sha(C / 'observations.jsonl'), 'main_gate_unchanged': True,
                    'decoding': {'batch_size': 4, 'do_sample': False, 'max_new_tokens': 16, 'max_time': 8, 'valid_seconds_below': 9.5},
                    'checkpoint': {a: {'path': (R / 'study' / a).relative_to(ROOT).as_posix(), 'seal': read(R / 'study' / a / 'CHECKPOINT_SEALED.json')} for a in plan['arms']}}
    (C / 'PLAN.json').write_text(json.dumps(registration, indent=2), encoding='utf-8')
    print(json.dumps({'new_responses': 672, 'main_gate_unchanged': True}))


def run():
    import gc
    import hashlib
    sys.path.insert(0, str(R.parent / '20261117_frozen_crossover'))
    from common import load, generate, verify as verify_model
    plan, registration = verify_control(); verify_model(); obs = observations()
    def digest(model):
        h = hashlib.sha256()
        for name, p in model.named_parameters():
            if p.requires_grad: h.update(name.encode()); h.update(p.detach().numpy().tobytes())
        return h.hexdigest()
    for arm in plan['arms']:
        path = C / (arm + '.jsonl'); assert not path.exists()
        tokenizer, model = load(arm, {'checkpoints': registration['checkpoint']}); model.eval(); before = digest(model)
        for start in range(0, len(obs), 4):
            batch = obs[start:start + 4]
            lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': o['prompt']}], tokenize=True, add_generation_prompt=True)) for o in batch]
            raw, seconds = generate(tokenizer, model, [o['prompt'] for o in batch])
            with path.open('a', encoding='utf-8') as stream:
                for o, text, length in zip(batch, raw, lengths):
                    stream.write(json.dumps({'id': o['id'], 'raw': text, 'prompt_tokens': length, 'padded_width': max(lengths), 'seconds_batch': seconds}) + '\n')
        after = digest(model); assert before == after
        (C / (arm + '_IDENTITY.json')).write_text(json.dumps({'before': before, 'after': after, 'errors': 0}))
        print(json.dumps({'arm': arm, 'responses': len(obs)}), flush=True); del model; gc.collect()
    names = ['PLAN.json', 'observations.jsonl'] + [a + suffix for a in plan['arms'] for suffix in ['.jsonl', '_IDENTITY.json']]
    (C / 'SEALED.json').write_text(json.dumps({'complete': True, 'files': {n: sha(C / n) for n in names}}, indent=2))


def audit():
    import re
    plan, registration = verify_control(); seal = read(C / 'SEALED.json')
    assert seal['complete'] and all(sha(C / n) == h for n, h in seal['files'].items())
    obs = observations(); main_obs, main_labels = cases(); third, labels = third_cases()
    label_map = {x['id']: x for x in labels + main_labels}; summaries = []; times = []
    def bit(row):
        found = re.findall(r'ANSWER:\s*([01])\b', row['raw'])
        return found[0] if len(found) == 1 and row['seconds_batch'] < 9.5 else None
    for arm in plan['arms']:
        raw = rows(C / (arm + '.jsonl')); assert len(raw) == 336
        main = rows(R / 'study' / arm / 'query_raw.jsonl'); separate = {r['id']: r for r in rows(R / 'study' / arm / 'third_raw.jsonl')}
        original_equal = third_equal = valid = matched_widths = original_bits = third_bits = 0
        invalid = collections.Counter(); pairs = collections.defaultdict(list); changes = []
        for i, (o, answer, before) in enumerate(zip(obs, raw, main)):
            assert answer['id'] == o['id']; label = label_map[o['id']]
            matched_widths += int(answer['padded_width'] == max(x['prompt_tokens'] for x in main[(i // 4) * 4:(i // 4) * 4 + 4]))
            assert answer['prompt_tokens'] == before['prompt_tokens']
            valid += bit(answer) is not None
            if o['vocab'] == 'original':
                assert before['id'] == answer['id']; original_equal += answer['raw'] == before['raw']
                original_bits += bit(answer) == bit(before)
                prior = before
            else:
                third_equal += answer['raw'] == separate[o['id']]['raw']
                third_bits += bit(answer) == bit(separate[o['id']]); prior = separate[o['id']]
                if answer['raw'] != separate[o['id']]['raw']: changes.append({'id': o['id'], 'separate': separate[o['id']]['raw'], 'matched': answer['raw']})
                if label['changing']: pairs[o['group']].append((o['side'], bit(answer) == str(label['target'])))
            for name, record in [('primary', prior), ('calibration', answer)]:
                invalid[o['vocab'] + '_' + name + '_invalid'] += int(bit(record) is None)
                invalid[o['vocab'] + '_' + name + '_timing_cutoff'] += int(record['seconds_batch'] >= 9.5)
            if i % 4 == 0:
                assert len({r['seconds_batch'] for r in raw[i:i + 4]}) == 1; times.append(answer['seconds_batch'])
        assert len(pairs) == 48 and all(len(v) == 2 and {x[0] for x in v} == {0, 1} for v in pairs.values())
        identity = read(C / (arm + '_IDENTITY.json')); assert identity['before'] == identity['after'] and identity['errors'] == 0
        summaries.append({'arm': arm, 'original_raw_identical': original_equal, 'original_total': 168,
                          'third_raw_identical_to_separate': third_equal, 'third_total': 168, 'third_changes': changes,
                          'original_parsed_equal': original_bits, 'third_parsed_equal_to_separate': third_bits,
                          'invalid_and_timing': dict(invalid),
                          'valid': valid, 'responses': 336, 'matched_padded_widths': matched_widths,
                          'third_paired_correct': sum(all(x[1] for x in v) for v in pairs.values()), 'changing_pairs': 48,
                          'numeric_control_passed': original_equal == 168 and matched_widths == 336 and valid == 336})
    result = {'summaries': summaries, 'new_responses': 672, 'new_updates': 0, 'generation_seconds': sum(times),
              'max_generation_seconds': max(times), 'errors': 0, 'main_gate_unchanged': True, 'whole_goal_complete': False}
    (C / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__':
    {'build': build, 'run': run, 'audit': audit}[sys.argv[1]]()
