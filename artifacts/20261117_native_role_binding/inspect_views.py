"""Measure complete role-local native inputs before attempting learned collaboration."""
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_native_roles'))
from session import RoleSession, SOURCE
from tau2.domains.telecom.environment import get_tasks


def main():
    assert not (R / 'INPUT_INVENTORY.json').exists()
    tasks = sorted(get_tasks(), key=lambda task: hashlib.sha256(('native-probe-17:' + task.id).encode()).hexdigest())
    task = tasks[0]
    session = RoleSession(task)
    views = {role: session.view(role) for role in ['assistant', 'user']}
    (R / 'initial_views.json').write_text(json.dumps(views, indent=2), encoding='utf-8')
    result = {'task_selection': 'minimum SHA256(native-probe-17:+task.id), no outcomes inspected',
              'task_id_outside_policy_input': task.id, 'source_manifest_sha256': hashlib.sha256((SOURCE / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
              'session_sha256': hashlib.sha256((R.parent / '20261117_native_roles/session.py').read_bytes()).hexdigest(),
              'views_sha256': hashlib.sha256((R / 'initial_views.json').read_bytes()).hexdigest(),
              'roles': {role: {'instructions_characters': len(view['instructions']), 'tool_count': len(view['tools']),
                               'serialized_view_characters': len(json.dumps(view)),
                               'tool_names': [tool['function']['name'] for tool in view['tools']]} for role, view in views.items()},
              'model_calls': 0, 'native_tools_executed': 0, 'whole_goal_complete': False}
    (R / 'INPUT_INVENTORY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
