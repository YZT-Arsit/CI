"""Audit original TRAIN presentations; do not pool them into confirmation."""
import collections
import json
import re
from data import F, R, NAMES, read, rows, sha, verified, render


def main():
    main_plan, plan = verified(); seal = read(F / 'SEALED.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(F / 'PLAN.json') and seal['parent_seal_sha256'] == sha(R / 'SEALED.json')
    assert all(sha(F / n) == h for n, h in seal['files'].items())
    obs = rows(F / 'observations.jsonl'); labels = rows(F / 'labels.jsonl'); pool = read(NAMES / 'palettes.json')['train']
    assert len(obs) == len(labels) == 144 and len({o['id'] for o in obs}) == 144
    for o, label in zip(obs, labels):
        assert o['id'] == label['id']; w = read(R / f"worlds_{o['seed']}.json")[o['world_id']]
        batch = rows(R / f"schedule_{o['seed']}.jsonl")[o['step']]; i = batch['ids'].index(o['world_id'])
        assert 0 <= o['step'] < 12 and o['name_seed'] == batch['names'][i] and label['target'] == batch['labels'][i]
        for mode in ['binary', 'proof']:
            text, valid = render(w, pool, o['name_seed'], mode)
            assert o['proof_prompt' if mode == 'proof' else 'prompt'] == text and label['valid_witnesses'] == valid
    summaries = []; times = []; total = 0
    for arm, config in main_plan['arms'].items():
        identity = read(F / 'outputs' / f'{arm}_IDENTITY.json'); source = R / 'study' / arm
        assert identity['errors'] == 0 and identity['before'] == identity['after'] == read(source / 'EVAL_IDENTITY.json')['before']
        assert identity['checkpoint']['seal'] == read(source / 'CHECKPOINT_SEALED.json')
        selected = [(o, l) for o, l in zip(obs, labels) if o['seed'] == config['seed']]; assert len(selected) == 48
        for mode in ['binary', 'proof']:
            raw = rows(F / 'outputs' / f'{arm}_{mode}.jsonl'); assert len(raw) == 48
            count = collections.Counter(); pairs = collections.defaultdict(list); length_groups = collections.defaultdict(list)
            for i, ((o, label), r) in enumerate(zip(selected, raw)):
                assert o['id'] == r['id']
                if mode == 'binary':
                    matches = re.findall(r'ANSWER:\s*([01])\b', r['raw']); value = int(matches[0]) if len(matches) == 1 else None
                else:
                    match = re.fullmatch(r'\s*PROOF:\s*([A-Za-z]+)\s*', r['raw']); value = match.group(1) if match else None
                if r['seconds_batch'] >= 9.5: value = None
                correct = value == label['target'] if mode == 'binary' else value in label['valid_witnesses']
                count.update(queries=1, correct=int(correct), invalid=int(value is None))
                count['positive_correct' if label['target'] else 'negative_correct'] += int(correct)
                if o['kind'] == 'switch':
                    pairs[o['group']].append((o['side'], correct)); length_groups[o['group']].append((r['prompt_tokens'], r['padded_width']))
                else: count.update(controls=1, controls_correct=int(correct))
                if i % 4 == 0:
                    batch = raw[i:i + 4]; assert len({v['seconds_batch'] for v in batch}) == 1
                    assert all(v['padded_width'] == max(x['prompt_tokens'] for x in batch) for v in batch); times.append(r['seconds_batch'])
            assert len(pairs) == 18 and all(len(v) == 2 and {x[0] for x in v} == {0, 1} for v in pairs.values())
            assert all(len(set(v)) == 1 for v in length_groups.values()) and count['controls'] == 12
            summaries.append({'arm': arm, 'seed': config['seed'], 'mode': mode, **dict(count),
                              'paired_correct': sum(all(x[1] for x in v) for v in pairs.values()), 'pairs': 18})
            total += 48
    assert total == 864
    result = {'summaries': summaries, 'new_responses': total, 'new_updates': 0, 'generation_seconds': sum(times),
              'max_generation_seconds': max(times), 'selected_train_only': True, 'errors': 0, 'whole_goal_complete': False}
    (F / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__': main()
