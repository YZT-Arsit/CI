import json
from pathlib import Path
import run_policy
import audit_policy
plan = audit_policy.verify()
assert run_policy.R == Path(__file__).resolve().parent and run_policy.runner.R == run_policy.PARENT
assert run_policy.evaluate.__code__.co_filename.endswith('20261117_action_binding/learned_evidence.py')
assert Path(run_policy.evaluate.__globals__['environment'].__code__.co_filename).resolve() == run_policy.ROOT / 'code/experiments/relation_solution_data.py'
assert audit_policy.replay.R == run_policy.R
print(json.dumps({'plan_sha256': run_policy.sha(run_policy.R / 'PLAN.json'), 'script_sha256': run_policy.sha(Path(__file__)),
                  'evaluate_source': run_policy.evaluate.__code__.co_filename, 'native_auditor_root': str(audit_policy.replay.R),
                  'model_loaded': False, 'model_calls': 0}))
