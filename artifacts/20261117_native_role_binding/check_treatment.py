"""Verify this follow-up changes prompt placement, not native observation/action routing."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
old = R.parent / '20261117_native_agent_probe'
plan = json.loads((R / 'PLAN.json').read_text())
prior = json.loads((old / 'PLAN.json').read_text())
keys = ['task_id_outside_policy_input', 'model', 'options', 'request_timeout_seconds',
        'warmup_timeout_seconds', 'max_model_decisions', 'context_safety_margin',
        'tokenizer', 'source_manifest_sha256', 'session_sha256', 'policy_pages']
assert all(plan[key] == prior[key] for key in keys)
files = ['run.py', 'transport.py', 'audit.py', 'initial_views.json', 'INPUT_INVENTORY.json']
assert all((R / name).read_bytes() == (old / name).read_bytes() for name in files)
assert all(hashlib.sha256((R / name).read_bytes()).hexdigest() == digest for name, digest in plan['source_hashes'].items())
assert plan['post_primary_diagnostic'] and plan['added_independent_tasks'] == 0
assert not (R / 'RUN_PLAN.json').exists()
result = {'unchanged_plan_keys': keys, 'byte_identical_files': files,
          'plan_source_hashes_match': True, 'model_calls_so_far': 0, 'new_training_updates': 0,
          'primary_output_seal_sha256': plan['primary_seal_sha256']}
(R / 'TREATMENT_CHECK.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
