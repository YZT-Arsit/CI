"""One predeclared format-only follow-up to the failed frozen reference."""
import collections
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent))
from learning_data import ROOT, read, rows, sha, verify


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


legacy = ROOT / 'artifacts/20261117_grounded_norm_probe/run.py'
request = module('norm_request', legacy).request
token_source = ROOT / 'artifacts/20261117_native_agent_probe/tokens.py'


def selected():
    obs = [o for o in rows(R.parent / 'observations.jsonl') if o['mode'] == 'zero_shot' and o['vocab'] == 'third']
    groups = {min(o['group'] for o in obs if o['kind'] == 'switch' and o['family'] == f and o['noise'] == n)
              for f in ['rewire', 'reverse_cycle'] for n in [0, 20]}
    groups |= {min(o['group'] for o in obs if o['kind'] == k) for k in ['known', 'empty_frontier', 'two_hop_only', 'direct']}
    result = [{**o, 'prompt': o['prompt'] + '\nOutput exactly one line: ANSWER: 0 or ANSWER: 1. Do not add any explanation.'} for o in obs if o['group'] in groups]; assert len(result) == 24
    return result


def build():
    verify(); assert not (R / 'PLAN.json').exists()
    previous = R.parent / 'capacity_reference'
    assert read(previous / 'SEALED.json')['complete']
    assert read(previous / 'RESULTS.json')['summaries'][0]['invalid'] == 24
    assert all(r['response']['done_reason'] == 'length' for r in rows(previous / 'responses.jsonl'))
    tokens = module('qwen_tokens', token_source)
    obs = []
    for o in selected():
        rendered, length = tokens.render(o['prompt']); assert length + 12 + 64 <= 2048
        obs.append({**o, 'rendered_prompt': rendered, 'prompt_tokens': length})
    (R / 'observations.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in obs), encoding='utf-8')
    model = next(r for r in request('tags')['models'] if r['name'] == 'qwen2.5:7b-instruct-q4_K_M')
    plan = {'model': model, 'main_plan_sha256': sha(R.parent / 'PLAN.json'), 'new_responses': 24, 'internal_rescorings': 0,
            'options': {'temperature': 0, 'seed': 17, 'num_ctx': 2048, 'num_predict': 12, 'num_gpu': 0, 'num_thread': 4},
            'request_timeout_seconds': 20, 'dependencies': {p.relative_to(ROOT).as_posix(): sha(p) for p in [legacy, token_source]},
            'files': {n: sha(R / n) for n in ['PLAN.md', 'reference.py', 'observations.jsonl']},
            'tokenizer': tokens.CONFIG, 'trigger': {n: sha(previous / n) for n in ['PLAN.json', 'SEALED.json', 'RESULTS.json']}}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8'); print(json.dumps({'queries': 24, 'max_prompt_tokens': max(o['prompt_tokens'] for o in obs)}))


def check():
    verify(); plan = read(R / 'PLAN.json')
    assert all(sha(R.parent / 'capacity_reference' / n) == h for n, h in plan['trigger'].items())
    assert plan['main_plan_sha256'] == sha(R.parent / 'PLAN.json')
    assert all(sha(R / n) == h for n, h in plan['files'].items())
    assert all(sha(ROOT / n) == h for n, h in plan['dependencies'].items())
    obs = rows(R / 'observations.jsonl'); assert [{k: v for k, v in o.items() if k not in ['rendered_prompt', 'prompt_tokens']} for o in obs] == selected()
    tokens = module('qwen_tokens_check', token_source)
    assert tokens.CONFIG == plan['tokenizer']
    assert all(tokens.render(o['prompt']) == (o['rendered_prompt'], o['prompt_tokens']) for o in obs)
    pairs = collections.defaultdict(list)
    for o in obs:
        if o['kind'] == 'switch': pairs[o['group'], o['view']].append(o['prompt_tokens'])
    assert len(pairs) == 8 and all(len(v) == 2 and len(set(v)) == 1 for v in pairs.values())
    return plan, obs


def run():
    plan, obs = check(); seal = read(R.parent / 'SEALED.json'); assert seal['complete']
    assert all(sha(R.parent / n) == h for n, h in seal['files'].items())
    assert not (R / 'RUN_PLAN.json').exists() and not request('ps')['models']
    model = plan['model']['name']; actual = next(r for r in request('tags')['models'] if r['name'] == model)
    assert actual['digest'] == plan['model']['digest']
    (R / 'RUN_PLAN.json').write_text(json.dumps({'plan_sha256': sha(R / 'PLAN.json'), 'main_seal_sha256': sha(R.parent / 'SEALED.json'), 'labels_read': False, 'retries': 0}))
    completed = 0; failure = None
    try:
        start = time.perf_counter()
        warmup = request('generate', {'model': model, 'prompt': '', 'stream': False, 'keep_alive': '2m', 'options': {**plan['options'], 'num_predict': 0}}, 60)
        (R / 'WARMUP.json').write_text(json.dumps({'seconds': time.perf_counter() - start, 'response': warmup}, indent=2))
        assert warmup.get('done') is True
        active = request('ps'); (R / 'LOADED_MODEL.json').write_text(json.dumps(active, indent=2))
        loaded = next(r for r in active['models'] if r['name'] == model)
        assert loaded['digest'] == actual['digest'] and loaded['size_vram'] == 0
        for o in obs:
            start = time.perf_counter()
            response = request('generate', {'model': model, 'prompt': o['rendered_prompt'], 'raw': True, 'stream': False,
                               'keep_alive': '2m', 'options': plan['options']}, plan['request_timeout_seconds'])
            elapsed = time.perf_counter() - start
            with (R / 'responses.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'id': o['id'], 'response': response, 'seconds': elapsed}) + '\n')
            completed += 1
            assert elapsed < 20 and response.get('prompt_eval_count') == o['prompt_tokens'] and response.get('done') is True
            if completed % 4 == 0: print(json.dumps({'completed': completed, 'seconds': elapsed}), flush=True)
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error), 'completed': completed}
        (R / 'FAILURE.json').write_text(json.dumps(failure, indent=2))
    finally:
        try:
            release = request('generate', {'model': model, 'keep_alive': 0}, 10); after = request('ps', timeout=5)
            (R / 'RELEASE_CHECK.json').write_text(json.dumps({'response': release, 'model_absent': all(r['name'] != model for r in after['models']), 'after': after}, indent=2))
        except Exception as error:
            (R / 'RELEASE_FAILURE.json').write_text(json.dumps({'message': str(error), 'fallback_keep_alive': '2m'}))
    files = [p for p in R.glob('*.json') if p.name != 'SEALED.json'] + list(R.glob('responses.jsonl'))
    (R / 'SEALED.json').write_text(json.dumps({'complete': completed == 24 and failure is None, 'completed_calls': completed, 'failure': failure,
        'labels_scored': False, 'files': {p.name: sha(p) for p in files}}, indent=2))
    print(json.dumps({'completed': completed, 'failure': failure}), flush=True)


def score(obs, labels, raw):
    stats = collections.Counter(); pairs = collections.defaultdict(list); groups = collections.defaultdict(list)
    for o, r in zip(obs, raw):
        assert o['id'] == r['id']; matches = re.findall(r'ANSWER:\s*([01])\b', r['raw'])
        bit = int(matches[0]) if r['valid_generation'] and len(matches) == 1 else None
        correct = bit == labels[o['id']]['target']; stats.update(queries=1, correct=int(correct), invalid=int(bit is None))
        if o['kind'] == 'switch': pairs[o['group'], o['view']].append(correct)
        else: stats.update(controls=1, controls_correct=int(correct))
    assert len(raw) == len(obs) == 24 and len(pairs) == 8 and all(len(v) == 2 for v in pairs.values())
    for (group, _), values in pairs.items(): groups[group].append(all(values))
    assert len(groups) == 4 and all(len(v) == 2 for v in groups.values())
    return {**stats, 'pairs': 8, 'paired_correct': sum(all(v) for v in pairs.values()), 'two_view_complete': sum(all(v) for v in groups.values()),
            'panel_passed': stats['correct'] == 24 and stats['invalid'] == 0}


def audit():
    plan, obs = check(); seal = read(R / 'SEALED.json'); assert all(sha(R / n) == h for n, h in seal['files'].items())
    assert seal['complete'], 'Incomplete reference: preserve failure and do not claim a full panel result.'
    run_plan = read(R / 'RUN_PLAN.json'); assert run_plan['plan_sha256'] == sha(R / 'PLAN.json') and run_plan['main_seal_sha256'] == sha(R.parent / 'SEALED.json')
    main_seal = read(R.parent / 'SEALED.json'); assert all(sha(R.parent / n) == h for n, h in main_seal['files'].items())
    labels = {r['id']: r for r in rows(R.parent / 'labels.jsonl')}; values = rows(R / 'responses.jsonl'); qwen = []
    assert len(values) == seal['completed_calls'] == 24
    for o, r in zip(obs, values):
        assert o['id'] == r['id'] and r['response']['prompt_eval_count'] == o['prompt_tokens'] and r['seconds'] < 20
        qwen.append({'id': r['id'], 'raw': r['response']['response'], 'valid_generation': r['response'].get('done') is True and r['response'].get('done_reason') != 'length'})
    summaries = [{'arm': 'qwen7b', **score(obs, labels, qwen)}]
    loaded = next(r for r in read(R / 'LOADED_MODEL.json')['models'] if r['name'] == plan['model']['name'])
    assert loaded['digest'] == plan['model']['digest'] and loaded['size_vram'] == 0 and read(R / 'RELEASE_CHECK.json')['model_absent']
    result = {'summaries': summaries, 'new_responses': 24, 'internal_rescorings': 0, 'new_updates': 0, 'errors': 0,
              'generation_seconds': sum(r['seconds'] for r in values), 'max_generation_seconds': max(r['seconds'] for r in values),
              'warmup_seconds': read(R / 'WARMUP.json')['seconds'], 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__': {'build': build, 'run': run, 'audit': audit}[sys.argv[1]]()
