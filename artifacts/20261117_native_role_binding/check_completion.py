"""Post-seal validity check: native state assertions can pass before any action."""
import hashlib
import json
from pathlib import Path
from transport import Transport, R
from tau2.domains.telecom.environment import get_tasks


def main():
    seal = json.loads((R / 'RESPONSES_SEALED.json').read_text())
    assert seal['terminal']
    assert all(hashlib.sha256((R / name).read_bytes()).hexdigest() == digest for name, digest in seal['files'].items())
    plan = json.loads((R / 'PLAN.json').read_text())
    task = next(task for task in get_tasks() if task.id == plan['task_id_outside_policy_input'])
    criteria = task.evaluation_criteria
    initial = Transport(task)
    before = [initial.session.env.run_env_assertion(a, raise_assertion_error=False) for a in criteria.env_assertions or []]
    events = list(map(json.loads, (R / 'events.jsonl').read_text().splitlines())) if (R / 'events.jsonl').exists() else []
    successful = [event for event in events if event['command']['kind'] == 'tool' and
                  event['result'].get('type') == 'tool' and event['result']['error'] is None]
    matched = []
    for action in criteria.actions or []:
        keys = list(action.arguments) if action.compare_args is None else action.compare_args
        matches = [event['index'] for event in successful if event['role'] == action.requestor and
                   event['command']['name'] == action.name and
                   all(key in event['command']['arguments'] and event['command']['arguments'][key] == action.arguments[key] for key in keys)]
        matched.append({'action_id': action.action_id, 'requestor': action.requestor, 'name': action.name,
                        'compare_args': keys, 'matching_successful_call_indices': matches})
    result = {'post_primary_validity_diagnostic': True, 'criteria': criteria.model_dump(mode='json'),
              'initial_native_state_assertions': before, 'initial_assertions_already_pass': bool(before) and all(before),
              'successful_native_tool_calls': len(successful), 'strict_reference_action_matches': matched,
              'all_reference_actions_matched': bool(matched) and all(row['matching_successful_call_indices'] for row in matched),
              'official_benchmark_score': None, 'full_task_success_established': False,
              'scope': 'Post-seal scorer coverage diagnosis. Strict declared-argument matching over successful native calls, not a faithful official evaluator or a necessary condition for all alternative plans.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (R / 'COMPLETION_AUDIT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
