"""Freeze one outcome-blind native task and validate role-local references."""
import hashlib
import json
import urllib.request
from pathlib import Path
from transport import Transport, RoleSession, R
from tokens import render, CONFIG
from tau2.domains.telecom.environment import get_tasks


def main():
    assert not (R / 'PLAN.json').exists()
    inventory = json.loads((R / 'INPUT_INVENTORY.json').read_text())
    task = next(task for task in get_tasks() if task.id == inventory['task_id_outside_policy_input'])
    transport = Transport(task)
    assert ''.join(transport.policy_pages) == transport.session.view('assistant')['instructions']
    raw = json.loads((R / 'initial_views.json').read_text())
    assert all(transport.session.view(role) == raw[role] for role in raw)
    observations = {role: transport.observation(role) for role in raw}
    lengths = {role: render(transport.prompt(role))[1] for role in raw}
    altered = task.model_copy(deep=True)
    altered.id = 'HIDDEN_TASK_CANARY'
    altered.description.notes = 'HIDDEN_EVALUATOR_CANARY'
    altered.evaluation_criteria.actions = []
    other = Transport(altered)
    assert all(other.observation(role) == observations[role] for role in observations)
    checks = 0
    for role in ['assistant', 'user']:
        for schema in transport.session.view(role)['tools']:
            result, same_role, done = transport.execute(role, {'kind': 'schema', 'name': schema['function']['name']})
            assert result == schema and same_role == role and not done
            checks += 1
    for index, page in enumerate(transport.policy_pages):
        result, same_role, done = transport.execute('assistant', {'kind': 'policy', 'page': index})
        assert result == {'page': index, 'text': page} and same_role == 'assistant' and not done
        checks += 1
    assert transport.session.counters()['submitted_tool_calls'] == 0
    fresh = Transport(task)
    before = fresh.observation('assistant')
    fresh.session.message('user', 'DROP_CANARY', deliver=False)
    assert fresh.observation('assistant') == before
    fresh.execute('user', {'kind': 'message', 'text': 'DELIVERY_CANARY'})
    assert fresh.observation('assistant')['history'][-1]['content'] == 'DELIVERY_CANARY'
    for role in ['user', 'assistant']:
        for kind in ['message', 'finish']:
            before_prompt = fresh.prompt(role)
            result, next_role, done = fresh.execute(role, {'kind': kind, 'text': ''})
            assert result == {'error': 'nonempty text required'} and next_role == role and not done
            assert fresh.prompt(role) != before_prompt
            assert fresh.observation(role)['history'][-1]['request'] == {'kind': kind, 'text': ''}
    models = json.load(urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=5))
    model = next(row for row in models['models'] if row['name'] == 'qwen2.5:7b-instruct-q4_K_M')
    plan = {'task_id_outside_policy_input': task.id, 'task_selection': inventory['task_selection'],
            'source_manifest_sha256': inventory['source_manifest_sha256'], 'session_sha256': inventory['session_sha256'],
            'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')},
            'model': model, 'options': {'temperature': 0, 'seed': 17, 'num_ctx': 4096, 'num_predict': 128, 'num_gpu': 0, 'num_thread': 4},
            'request_timeout_seconds': 40, 'warmup_timeout_seconds': 60, 'max_model_decisions': 32,
            'context_safety_margin': 64, 'tokenizer': {k: CONFIG[k] for k in ['official_repo', 'official_revision', 'tokenizer_files']},
            'initial_prompt_tokens': lengths, 'policy_pages': len(transport.policy_pages),
            'reference_isolation_checks': checks, 'page_reconstruction_exact': True,
            'metadata_noninterference': True, 'message_transport_checks': True,
            'training_updates': 0, 'runner_reads_evaluation_criteria': False,
            'scope': 'Post-primary single-task role-conditioning diagnosis: same task and model, replace generic Qwen system message with an explicit participant-role and history-boundary instruction. Observation/history stay in user message. No independent case,training or native policy change; not an official benchmark or32-step result.',
            'primary_plan_sha256': hashlib.sha256((R.parent / '20261117_native_agent_probe/PLAN.json').read_bytes()).hexdigest(),
            'primary_seal_sha256': hashlib.sha256((R.parent / '20261117_native_agent_probe/RESPONSES_SEALED.json').read_bytes()).hexdigest(),
            'added_independent_tasks': 0, 'post_primary_diagnostic': True,
            'whole_goal_complete': False}
    assert max(lengths.values()) + 128 + 64 < 4096
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({k: plan[k] for k in ['initial_prompt_tokens', 'policy_pages', 'reference_isolation_checks', 'max_model_decisions']}))


if __name__ == '__main__':
    main()
