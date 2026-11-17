"""Replay sealed model commands and only then inspect native task assertions."""
import hashlib
import json
from pathlib import Path
from transport import Transport, R
from tokens import render
from tau2.domains.telecom.environment import get_tasks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_stop(raw, plan):
    response = raw['response']
    if raw['seconds'] > plan['request_timeout_seconds']:
        return 'response_wall_time_limit'
    if response.get('prompt_eval_count') != raw['prompt_tokens']:
        return 'token_count_mismatch'
    if response.get('done') is not True or response.get('done_reason') == 'length':
        return 'incomplete_response'
    try:
        command = json.loads(response['response'])
        assert isinstance(command, dict) and command.get('kind') in ['message', 'schema', 'policy', 'tool', 'finish']
    except (ValueError, AssertionError):
        return 'invalid_command_format'
    return None


def main():
    plan = json.loads((R / 'PLAN.json').read_text())
    run = json.loads((R / 'RUN_PLAN.json').read_text())
    seal = json.loads((R / 'RESPONSES_SEALED.json').read_text())
    assert run['plan_sha256'] == sha(R / 'PLAN.json') and run['source_sha256'] == sha(R / 'run.py')
    assert all(sha(R / name) == digest for name, digest in plan['source_hashes'].items())
    assert all(sha(R / name) == digest for name, digest in seal['files'].items())
    assert seal['terminal'] and not seal['labels_scored'] and not run['reads_evaluation_criteria']
    source = R.parent / '20261117_action_binding/tau2_source'
    assert sha(source / 'SOURCE_MANIFEST.json') == plan['source_manifest_sha256']
    manifest = json.loads((source / 'SOURCE_MANIFEST.json').read_text())
    for item in manifest['files']:
        raw = (source / item['path']).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == item['git_blob_sha']
    assert sha(R.parent / '20261117_native_roles/session.py') == plan['session_sha256']
    tasks = get_tasks()
    selected = min(tasks, key=lambda task: hashlib.sha256(('native-probe-17:' + task.id).encode()).hexdigest())
    assert selected.id == plan['task_id_outside_policy_input']
    transport = Transport(selected)
    raws = list(map(json.loads, (R / 'raw.jsonl').read_text().splitlines())) if (R / 'raw.jsonl').exists() else []
    events = list(map(json.loads, (R / 'events.jsonl').read_text().splitlines())) if (R / 'events.jsonl').exists() else []
    by_index = {row['index']: row for row in events}
    assert len(by_index) == len(events)
    assert [event['index'] for event in events] == list(range(len(events)))
    assert len(events) <= len(raws) <= len(events) + 1
    end = json.loads((R / 'END.json').read_text())
    assert len(raws) <= end['request_attempts'] <= plan['max_model_decisions']
    assert end['request_attempts'] <= len(raws) + 1
    role = 'user'
    done = False
    for index, raw in enumerate(raws):
        assert not done and raw['index'] == index and raw['role'] == role
        prompt = transport.prompt(role)
        rendered, length = render(prompt)
        assert raw['prompt'] == prompt and raw['rendered_prompt'] == rendered and raw['prompt_tokens'] == length
        assert length + plan['options']['num_predict'] + plan['context_safety_margin'] <= plan['options']['num_ctx']
        if index not in by_index:
            assert index == len(raws) - 1
            reason = response_stop(raw, plan)
            if reason is not None:
                assert end['stop_reason'] == reason
            else:
                assert end['stop_reason'] == 'request_or_execution_failure'
                failure = json.loads((R / 'FAILURE.json').read_text())
                try:
                    transport.execute(role, json.loads(raw['response']['response']))
                except Exception as error:
                    assert type(error).__name__ == failure['type'] and str(error) == failure['message']
                else:
                    raise AssertionError('unrecorded command executed successfully')
            break
        assert response_stop(raw, plan) is None
        command = json.loads(raw['response']['response'])
        event = by_index[index]
        assert event['role'] == role and event['command'] == command
        assert event['before'] == transport.session.counters()
        result, next_role, done = transport.execute(role, command)
        assert json.loads(json.dumps(result)) == event['result']
        assert next_role == event['next_role'] and done == event['done'] and event['after'] == transport.session.counters()
        role = next_role
    assert end['completed_model_calls'] == len(raws) and end['reference_lookups'] == len(transport.references)
    assert end['native_counters'] == transport.session.counters() and not end['native_assertions_evaluated']
    if end['stop_reason'] == 'context_limit_before_request':
        assert not done and len(events) == len(raws) == end['request_attempts'] < plan['max_model_decisions']
        context = json.loads((R / 'CONTEXT_LIMIT.json').read_text())
        assert context['index'] == len(raws) and context['silent_truncation'] is False
        assert context['prompt'] == transport.prompt(role) and context['role'] == role
        assert context['prompt_tokens'] == render(context['prompt'])[1]
        assert context['prompt_tokens'] + plan['options']['num_predict'] + plan['context_safety_margin'] > plan['options']['num_ctx']
    elif end['stop_reason'] == 'model_finish':
        assert done and len(events) == len(raws) == end['request_attempts']
        assert events[-1]['command']['kind'] == 'finish'
    elif end['stop_reason'] == 'decision_cap':
        assert not done and len(events) == len(raws) == end['request_attempts'] == plan['max_model_decisions']
    elif end['stop_reason'] == 'request_or_execution_failure':
        assert not done
        failure = json.loads((R / 'FAILURE.json').read_text())
        assert failure['attempts'] == end['request_attempts'] and failure['completed'] == len(raws)
        assert len(raws) == len(events) + 1 or end['request_attempts'] == len(raws) + 1 or end['request_attempts'] == 0
    else:
        assert len(raws) == len(events) + 1 and end['request_attempts'] == len(raws) and not done
        assert response_stop(raws[-1], plan) == end['stop_reason']
    assertions = selected.evaluation_criteria.env_assertions or []
    results = [transport.session.env.run_env_assertion(assertion, raise_assertion_error=False) for assertion in assertions]
    result = {'sealed_raw_calls_replayed': len(raws), 'model_commands_replayed': len(events),
              'stop_reason': end['stop_reason'], 'reference_lookups': len(transport.references),
              'native_counters': transport.session.counters(), 'native_state_assertions': results,
              'all_native_state_assertions_pass': bool(results) and all(results),
              'all_commands_from_raw_model_outputs': True, 'oracle_action_interventions': 0,
              'max_request_seconds': max((row['seconds'] for row in raws), default=None),
              'source_files_verified': len(manifest['files']), 'auditor_sha256': sha(Path(__file__)),
              'full_norm_compliance_established': False, '32_meaningful_collaboration_steps_established': False,
              'whole_goal_complete': False, 'scope': plan['scope']}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
