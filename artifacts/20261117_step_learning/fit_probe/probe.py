"""Frozen TRAIN-fit diagnosis with deterministic schedule-only selection."""
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent))
from learning_data import ROOT, NAMES, read, rows, sha, verify, training_schedule, train_worlds, prompt, label


def cases():
    data = train_worlds(); pools = read(NAMES / 'palettes.json'); result = []
    for seed in [17, 29, 41]:
        selected = []; control_taken = False
        for batch in training_schedule(seed)[-8:]:
            if batch['kind'] == 'control':
                if control_taken: continue
                control_taken = True; count = 4
            else: count = 2
            selected.append((batch, count))
        for view in ['seen', 'renamed_train', 'third']:
            for index, (batch, count) in enumerate(selected):
                for j in range(count):
                    key = batch['ids'][j]; world = data[key]
                    name_seed = batch['names'][j] if view == 'seen' else 470000000 + seed * 1000 + index
                    pool = pools['new' if view == 'third' else 'train']
                    result.append({'id': f'{seed}_{view}_{index}_{j}', 'seed': seed, 'view': view,
                                   'world_id': key, 'step': batch['step'], 'name_seed': name_seed,
                                   'kind': batch['kind'], 'pair': index if batch['kind'] == 'switch' else None,
                                   'target': label(world), 'prompt': prompt(world, pool, name_seed)})
    assert len(result) == 144
    return result


def build():
    verify(); assert not (R / 'PLAN.json').exists()
    (R / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in cases()), encoding='utf-8')
    plan = {'main_plan_sha256': sha(R.parent / 'PLAN.json'), 'new_responses': 288, 'new_updates': 0,
            'files': {n: sha(R / n) for n in ['PLAN.md', 'probe.py', 'cases.jsonl']}}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan))


def check():
    verify(); plan = read(R / 'PLAN.json')
    assert plan['main_plan_sha256'] == sha(R.parent / 'PLAN.json')
    assert all(sha(R / n) == h for n, h in plan['files'].items())
    assert cases() == rows(R / 'cases.jsonl')
    return plan


def run():
    check(); main_seal = read(R.parent / 'SEALED.json')
    assert main_seal['complete'] and all(sha(R.parent / n) == h for n, h in main_seal['files'].items())
    sys.path[:0] = [str(R.parents[1] / '20261117_frozen_crossover')]
    from common import load, generate, verify as verify_model
    spec = importlib.util.spec_from_file_location('local_step_worker', R.parent / 'run.py')
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    verify_model(); output = R / 'outputs'; assert not output.exists(); output.mkdir()
    observations = rows(R / 'cases.jsonl')
    for arm in ['initial', 'seed17', 'seed29', 'seed41']:
        source = R.parent / 'learning' / arm
        cp = {'path': source.relative_to(ROOT).as_posix(), 'seal': read(source / 'CHECKPOINT_SEALED.json')}
        tokenizer, model = load('start', {'checkpoints': {'start': cp}}); model.eval(); before = worker.digest(model)
        selected = observations if arm == 'initial' else [o for o in observations if o['seed'] == int(arm[4:])]
        for start in range(0, len(selected), 4):
            batch = selected[start:start + 4]
            lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': o['prompt']}], tokenize=True, add_generation_prompt=True)) for o in batch]
            raw, seconds = generate(tokenizer, model, [o['prompt'] for o in batch])
            for o, text, length in zip(batch, raw, lengths):
                worker.append(output / (arm + '.jsonl'), {'id': o['id'], 'raw': text, 'prompt_tokens': length,
                              'padded_width': max(lengths), 'seconds_batch': seconds})
        after = worker.digest(model); assert before == after
        (output / (arm + '_identity.json')).write_text(json.dumps({'before': before, 'after': after, 'checkpoint': cp}), encoding='utf-8')
        print(json.dumps({'arm': arm, 'responses': len(selected)}), flush=True)
        del model
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'main_seal_sha256': sha(R.parent / 'SEALED.json'), 'files': {p.relative_to(R).as_posix(): sha(p) for p in output.iterdir()}}, indent=2), encoding='utf-8')


def audit():
    check(); seal = read(R / 'SEALED.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json') and seal['main_seal_sha256'] == sha(R.parent / 'SEALED.json')
    assert all(sha(R / n) == h for n, h in seal['files'].items())
    observations = rows(R / 'cases.jsonl'); summaries = []; times = []
    for arm in ['initial', 'seed17', 'seed29', 'seed41']:
        identity = read(R / 'outputs' / (arm + '_identity.json')); assert identity['before'] == identity['after']
        source = ROOT / identity['checkpoint']['path']; assert identity['checkpoint']['seal'] == read(source / 'CHECKPOINT_SEALED.json')
        assert all(sha(source / 'adapter' / n) == h for n, h in identity['checkpoint']['seal']['adapter_hashes'].items())
        selected = observations if arm == 'initial' else [o for o in observations if o['seed'] == int(arm[4:])]
        raw = rows(R / 'outputs' / (arm + '.jsonl')); assert len(raw) == len(selected)
        stats = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(list)
        for i, (o, answer) in enumerate(zip(selected, raw)):
            assert o['id'] == answer['id']; matches = re.findall(r'ANSWER:\s*([01])\b', answer['raw'])
            bit = int(matches[0]) if len(matches) == 1 and answer['seconds_batch'] < 9.5 else None
            correct = bit == o['target']; key = (o['seed'], o['view'])
            stats[key].update(queries=1, correct=int(correct), invalid=int(bit is None),
                              controls=int(o['kind'] == 'control'), controls_correct=int(o['kind'] == 'control' and correct),
                              false_positive=int(bit == 1 and o['target'] == 0), false_negative=int(bit == 0 and o['target'] == 1))
            if o['kind'] == 'switch': pairs[(key, o['pair'])].append((correct, answer['prompt_tokens'], answer['padded_width'], o['target']))
            if i % 4 == 0:
                batch = raw[i:i + 4]; assert len({r['seconds_batch'] for r in batch}) == 1
                assert all(r['padded_width'] == max(x['prompt_tokens'] for x in batch) for r in batch)
                times.append(answer['seconds_batch'])
        for (key, _), values in pairs.items():
            assert len(values) == 2 and {v[3] for v in values} == {0, 1}
            assert len({v[1] for v in values}) == len({v[2] for v in values}) == 1
            stats[key].update(pairs=1, paired_correct=int(all(v[0] for v in values)))
        for (seed, view), counts in stats.items():
            assert counts['queries'] == 16 and counts['pairs'] == 6 and counts['controls'] == 4
            summaries.append({'arm': arm, 'schedule': seed, 'view': view, **counts})
    result = {'summaries': summaries, 'new_responses': 288, 'new_updates': 0, 'generation_seconds': sum(times),
              'max_generation_seconds': max(times), 'errors': 0, 'main_gate_unchanged': True, 'whole_goal_complete': False}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__': {'build': build, 'run': run, 'audit': audit}[sys.argv[1]]()
