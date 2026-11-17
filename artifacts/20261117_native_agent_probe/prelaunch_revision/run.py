"""Run only model-selected role-local commands; seal before native evaluation."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from transport import Transport, R
from tokens import render
from tau2.domains.telecom.environment import get_tasks


def request(endpoint, payload=None, timeout=40):
    req = urllib.request.Request('http://127.0.0.1:11434/api/' + endpoint,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def append(name, row):
    with (R / name).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row) + '\n')


def main():
    assert not (R / 'RUN_PLAN.json').exists()
    plan = json.loads((R / 'PLAN.json').read_text())
    assert all(hashlib.sha256((R / name).read_bytes()).hexdigest() == digest for name, digest in plan['source_hashes'].items())
    tags = request('tags')
    model = plan['model']['name']
    actual = next(row for row in tags['models'] if row['name'] == model)
    assert actual['digest'] == plan['model']['digest']
    assert not request('ps')['models'], 'Defer while a user model is loaded.'
    task = next(task for task in get_tasks() if task.id == plan['task_id_outside_policy_input'])
    transport = Transport(task)
    (R / 'RUN_PLAN.json').write_text(json.dumps({'plan_sha256': hashlib.sha256((R / 'PLAN.json').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'model_digest': actual['digest'],
        'reads_evaluation_criteria': False, 'retrieval_is_model_selected': True, 'retry_count': 0}, indent=2))
    role, stop, completed, attempts = 'user', 'decision_cap', 0, 0
    try:
        start = time.perf_counter()
        warmup = request('generate', {'model': model, 'prompt': '', 'stream': False, 'keep_alive': '2m',
                          'options': {**plan['options'], 'num_predict': 0}}, plan['warmup_timeout_seconds'])
        (R / 'WARMUP.json').write_text(json.dumps({'seconds': time.perf_counter() - start, 'response': warmup}, indent=2))
        loaded = request('ps')
        (R / 'LOADED_MODEL.json').write_text(json.dumps(loaded, indent=2))
        active = next(row for row in loaded['models'] if row['name'] == model)
        assert active['digest'] == actual['digest'] and active['size_vram'] == 0
        for index in range(plan['max_model_decisions']):
            prompt = transport.prompt(role)
            rendered, length = render(prompt)
            if length + plan['options']['num_predict'] + plan['context_safety_margin'] > plan['options']['num_ctx']:
                stop = 'context_limit_before_request'
                (R / 'CONTEXT_LIMIT.json').write_text(json.dumps({'index': index, 'role': role, 'prompt_tokens': length,
                                                               'prompt': prompt, 'silent_truncation': False}, indent=2))
                break
            attempts += 1
            start = time.perf_counter()
            response = request('generate', {'model': model, 'prompt': rendered, 'raw': True, 'stream': False,
                               'keep_alive': '2m', 'options': plan['options']}, plan['request_timeout_seconds'])
            elapsed = time.perf_counter() - start
            append('raw.jsonl', {'index': index, 'role': role, 'prompt': prompt, 'rendered_prompt': rendered,
                                'prompt_tokens': length, 'response': response, 'seconds': elapsed})
            completed += 1
            if elapsed > plan['request_timeout_seconds']:
                stop = 'response_wall_time_limit'; break
            if response.get('prompt_eval_count') != length:
                stop = 'token_count_mismatch'; break
            if response.get('done') is not True or response.get('done_reason') == 'length':
                stop = 'incomplete_response'; break
            try:
                command = json.loads(response['response'])
                assert isinstance(command, dict) and command.get('kind') in ['message', 'schema', 'policy', 'tool', 'finish']
            except (ValueError, AssertionError):
                stop = 'invalid_command_format'; break
            before = transport.session.counters()
            result, next_role, done = transport.execute(role, command)
            append('events.jsonl', {'index': index, 'role': role, 'command': command, 'result': result,
                                   'next_role': next_role, 'done': done, 'before': before,
                                   'after': transport.session.counters()})
            print(json.dumps({'index': index, 'role': role, 'kind': command['kind'], 'seconds': elapsed}), flush=True)
            role = next_role
            if done:
                stop = 'model_finish'; break
    except Exception as error:
        stop = 'request_or_execution_failure'
        (R / 'FAILURE.json').write_text(json.dumps({'type': type(error).__name__, 'message': str(error),
                                                  'attempts': attempts, 'completed': completed}, indent=2))
    finally:
        try:
            result = request('generate', {'model': model, 'keep_alive': 0}, 10)
            after = request('ps', timeout=5)
            (R / 'RELEASE_CHECK.json').write_text(json.dumps({'unload_response': result,
                'model_absent': all(row['name'] != model for row in after['models']), 'loaded_models_after': after}, indent=2))
        except Exception as error:
            (R / 'RELEASE_FAILURE.json').write_text(json.dumps({'error': str(error), 'fallback': '2m expiry'}))
    end = {'stop_reason': stop, 'request_attempts': attempts, 'completed_model_calls': completed,
           'reference_lookups': len(transport.references), 'native_counters': transport.session.counters(),
           'native_assertions_evaluated': False, 'whole_goal_complete': False}
    (R / 'END.json').write_text(json.dumps(end, indent=2))
    names = [name for name in ['raw.jsonl', 'events.jsonl', 'END.json', 'CONTEXT_LIMIT.json', 'FAILURE.json'] if (R / name).exists()]
    (R / 'RESPONSES_SEALED.json').write_text(json.dumps({'terminal': True, 'labels_scored': False,
        'files': {name: hashlib.sha256((R / name).read_bytes()).hexdigest() for name in names}}, indent=2))
    print(json.dumps(end), flush=True)


if __name__ == '__main__':
    main()
