"""Small negative controls for response rejection, without model/native calls."""
import copy
import hashlib
import json
from pathlib import Path
from audit import response_stop

R = Path(__file__).resolve().parent
plan = json.loads((R / 'PLAN.json').read_text())
valid = {'prompt_tokens': 9, 'seconds': 1., 'response': {'prompt_eval_count': 9,
         'done': True, 'done_reason': 'stop', 'response': '{"kind":"message","text":"hello"}'}}
checks = []
for field, value, expected in [('seconds', 41., 'response_wall_time_limit'),
                              ('prompt_eval_count', 8, 'token_count_mismatch'),
                              ('done', False, 'incomplete_response'),
                              ('done_reason', 'length', 'incomplete_response'),
                              ('response', 'not JSON', 'invalid_command_format'),
                              ('response', '[]', 'invalid_command_format'),
                              ('response', '{"kind":"invented"}', 'invalid_command_format')]:
    raw = copy.deepcopy(valid)
    (raw if field == 'seconds' else raw['response'])[field] = value
    assert response_stop(raw, plan) == expected
    checks.append({'field': field, 'expected': expected, 'pass': True})
assert response_stop(valid, plan) is None
assert all(hashlib.sha256((R / name).read_bytes()).hexdigest() == digest for name, digest in plan['source_hashes'].items())
result = {'negative_controls': checks, 'valid_response_accepted': True, 'plan_source_hashes_match': True,
          'prelaunch_only': True, 'model_calls': 0, 'native_tool_calls': 0,
          'auditor_sha256': hashlib.sha256((R / 'audit.py').read_bytes()).hexdigest()}
(R / 'PRELAUNCH_CHECKS.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
