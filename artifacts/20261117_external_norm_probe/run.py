"""CPU-only free generation; this worker never opens the label file."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

R = Path(__file__).resolve().parent


def request(endpoint, payload=None, timeout=25):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request('http://127.0.0.1:11434/api/' + endpoint, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    assert not (R / 'RUN_PLAN.json').exists()
    plan = json.loads((R / 'PLAN.json').read_text())
    scoring = json.loads((R / 'SCORING_PLAN.json').read_text())
    assert scoring['auditor_sha256'] == hashlib.sha256((R / 'audit.py').read_bytes()).hexdigest()
    assert hashlib.sha256((R / 'observations.jsonl').read_bytes()).hexdigest() == plan['data_hashes']['observations.jsonl']
    observations = list(map(json.loads, (R / 'observations.jsonl').read_text().splitlines()))
    assert len(observations) == plan['presentations']
    model = plan['model']['name']
    actual = next(row for row in request('tags')['models'] if row['name'] == model)
    assert actual['digest'] == plan['model']['digest']
    active = request('ps')
    assert not active['models'], 'Existing user-loaded model: defer without altering it.'
    assert all(len(row['prompt'].encode()) < 1500 for row in observations)
    runtime = {'plan_sha256': hashlib.sha256((R / 'PLAN.json').read_bytes()).hexdigest(),
               'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'builder_sha256': hashlib.sha256((R / 'build.py').read_bytes()).hexdigest(),
               'auditor_sha256': hashlib.sha256((R / 'audit.py').read_bytes()).hexdigest(),
               'scoring_plan_sha256': hashlib.sha256((R / 'SCORING_PLAN.json').read_bytes()).hexdigest(),
               'initial_loaded_models': active, 'model_digest': actual['digest'],
               'reads_labels': False, 'retry_count': 0, 'oracle_at_runtime': False}
    (R / 'RUN_PLAN.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    completed = 0
    failure = None
    try:
        start = time.perf_counter()
        warmup = request('generate', {'model': model, 'prompt': '', 'stream': False,
                         'keep_alive': '2m', 'options': {**plan['options'], 'num_predict': 0}}, plan['warmup_timeout_seconds'])
        (R / 'WARMUP.json').write_text(json.dumps({'seconds': time.perf_counter() - start, 'response': warmup}, indent=2))
        assert warmup.get('done') is True
        loaded = request('ps')
        (R / 'LOADED_MODEL.json').write_text(json.dumps(loaded, indent=2))
        active_model = next(row for row in loaded['models'] if row['name'] == model)
        assert active_model['digest'] == actual['digest'] and active_model['size_vram'] == 0
        for row in observations:
            start = time.perf_counter()
            payload = {'model': model, 'messages': [{'role': 'user', 'content': row['prompt']}],
                       'stream': False, 'keep_alive': '2m', 'options': plan['options']}
            response = request('chat', payload, plan['request_timeout_seconds'])
            elapsed = time.perf_counter() - start
            if elapsed > plan['request_timeout_seconds']:
                (R / 'OVERTIME_RESPONSE.json').write_text(json.dumps({**row, 'response': response, 'seconds': elapsed}, indent=2))
                raise TimeoutError('Completed response exceeded wall-time acceptance limit; retained separately, no retry.')
            assert response.get('done') is True
            assert response.get('prompt_eval_count', 9999) < plan['options']['num_ctx'] - 64
            record = {**row, 'response': response, 'seconds': elapsed}
            with (R / 'responses.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            completed += 1
            if completed % 8 == 0:
                print(json.dumps({'completed': completed, 'seconds': elapsed}), flush=True)
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error), 'completed': completed}
        (R / 'FAILURE.json').write_text(json.dumps(failure, indent=2))
    finally:
        # This run started with no loaded models; unload only its own model.
        try:
            release = request('generate', {'model': model, 'keep_alive': 0}, 10)
            (R / 'RELEASE.json').write_text(json.dumps(release, indent=2))
            after = request('ps', timeout=5)
            (R / 'RELEASE_CHECK.json').write_text(json.dumps({'model_absent': all(row['name'] != model for row in after['models']),
                                                            'loaded_models_after': after}, indent=2))
        except Exception as error:
            (R / 'RELEASE_FAILURE.json').write_text(json.dumps({'error': str(error), 'fallback': '2m keep_alive expiry'}))
    paths = ['observations.jsonl'] + [name for name in ['responses.jsonl', 'OVERTIME_RESPONSE.json'] if (R / name).exists()]
    seal = {'complete': completed == plan['presentations'] and failure is None, 'completed_calls': completed,
            'labels_scored': False, 'failure': failure,
            'files': {name: hashlib.sha256((R / name).read_bytes()).hexdigest() for name in paths}}
    (R / 'RESPONSES_SEALED.json').write_text(json.dumps(seal, indent=2))
    print(json.dumps(seal), flush=True)


if __name__ == '__main__':
    main()
