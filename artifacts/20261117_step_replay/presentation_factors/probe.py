"""Frozen2x2presentation experiment; no training or primary-gate changes."""
import collections
import importlib.util
import json
import re
import sys
from factor_data import R, ROOT, read, rows, sha, check


def run():
    plan = check(); main = read(R.parent / 'SEALED.json')
    assert main['complete'] and all(sha(R.parent / n) == h for n, h in main['files'].items())
    sys.path.insert(0, str(R.parents[1] / '20261117_frozen_crossover'))
    from common import load, generate, verify as verify_model
    spec = importlib.util.spec_from_file_location('rehearsal_worker', R.parent / 'run.py')
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    verify_model(); output = R / 'outputs'; assert not output.exists(); output.mkdir(); obs = rows(R / 'observations.jsonl')
    for arm, seed in plan['arms'].items():
        source = R.parent / 'study' / arm; cp = {'path': source.relative_to(ROOT).as_posix(), 'seal': read(source / 'CHECKPOINT_SEALED.json')}
        tokenizer, model = load('start', {'checkpoints': {'start': cp}}); model.eval(); before = worker.digest(model)
        selected = obs if seed is None else [o for o in obs if o['seed'] == seed]
        for start in range(0, len(selected), 4):
            batch = selected[start:start + 4]
            lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': o['prompt']}], tokenize=True, add_generation_prompt=True)) for o in batch]
            raw, seconds = generate(tokenizer, model, [o['prompt'] for o in batch])
            for o, text, length in zip(batch, raw, lengths):
                worker.append(output / (arm + '.jsonl'), {'id': o['id'], 'raw': text, 'prompt_tokens': length, 'padded_width': max(lengths), 'seconds_batch': seconds})
        after = worker.digest(model); assert before == after
        (output / (arm + '_identity.json')).write_text(json.dumps({'before': before, 'after': after, 'checkpoint': cp}), encoding='utf-8')
        print(json.dumps({'arm': arm, 'responses': len(selected)}), flush=True); del model
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'), 'main_seal_sha256': sha(R.parent / 'SEALED.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in output.iterdir()}}, indent=2), encoding='utf-8')


def audit():
    plan = check(); seal = read(R / 'SEALED.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json') and seal['main_seal_sha256'] == sha(R.parent / 'SEALED.json')
    assert all(sha(R / n) == h for n, h in seal['files'].items())
    obs = rows(R / 'observations.jsonl'); labels = {r['id']: r['target'] for r in rows(R / 'labels.jsonl')}
    summaries = []; times = []; responses = 0
    for arm, seed in plan['arms'].items():
        identity = read(R / 'outputs' / (arm + '_identity.json')); assert identity['before'] == identity['after']
        source = ROOT / identity['checkpoint']['path']; assert identity['checkpoint']['seal'] == read(source / 'CHECKPOINT_SEALED.json')
        assert all(sha(source / 'adapter' / n) == h for n, h in identity['checkpoint']['seal']['adapter_hashes'].items())
        selected = obs if seed is None else [o for o in obs if o['seed'] == seed]
        raw = rows(R / 'outputs' / (arm + '.jsonl')); assert len(raw) == len(selected); responses += len(raw)
        stats = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(list)
        for i, (o, r) in enumerate(zip(selected, raw)):
            assert o['id'] == r['id']; matches = re.findall(r'ANSWER:\s*([01])\b', r['raw'])
            bit = int(matches[0]) if len(matches) == 1 and r['seconds_batch'] < 9.5 else None
            target = labels[o['id']]; correct = bit == target; key = o['seed'], o['view']
            stats[key].update(queries=1, correct=int(correct), invalid=int(bit is None), controls=int(o['kind'] == 'control'),
                              controls_correct=int(o['kind'] == 'control' and correct), false_positive=int(bit == 1 and target == 0), false_negative=int(bit == 0 and target == 1))
            if o['kind'] == 'switch': pairs[key, o['pair']].append((correct, r['prompt_tokens'], r['padded_width'], target))
            if i % 4 == 0:
                batch = raw[i:i + 4]; assert len({b['seconds_batch'] for b in batch}) == 1
                assert all(b['padded_width'] == max(v['prompt_tokens'] for v in batch) for b in batch); times.append(r['seconds_batch'])
        for (key, _), values in pairs.items():
            assert len(values) == 2 and {v[3] for v in values} == {0, 1} and len({v[1] for v in values}) == len({v[2] for v in values}) == 1
            stats[key].update(pairs=1, paired_correct=int(all(v[0] for v in values)))
        for (panel, view), counts in stats.items():
            assert counts['queries'] == 16 and counts['pairs'] == 6 and counts['controls'] == 4
            summaries.append({'arm': arm, 'schedule': panel, 'view': view, **counts})
    assert responses == 576
    result = {'summaries': summaries, 'new_responses': responses, 'new_updates': 0, 'errors': 0, 'generation_seconds': sum(times),
              'max_generation_seconds': max(times), 'main_gate_unchanged': True, 'whole_goal_complete': False}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__': {'run': run, 'audit': audit}[sys.argv[1]]()
