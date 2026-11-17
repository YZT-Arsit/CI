import json
from pathlib import Path
import run_fit
import audit_fit
parent, plan = audit_fit.verified()
assert run_fit.F == Path(__file__).resolve().parent and run_fit.runner.R == run_fit.R
assert run_fit.w.R == run_fit.data.PARENT
assert Path(run_fit.w.forward_targets.__code__.co_filename).resolve() == run_fit.R.parent / '20261117_action_binding/learned_evidence.py'
prior = run_fit.read(run_fit.F / 'prelaunch_import_revision/PLAN.json')
assert prior['data'] == plan['data'] and prior['arms'] == plan['arms']
print(json.dumps({'plan_sha256': run_fit.sha(run_fit.F / 'PLAN.json'), 'script_sha256': run_fit.sha(Path(__file__)),
                  'data_and_arms_unchanged_from_import_revision': True, 'worker_root': str(run_fit.w.R), 'model_loaded': False, 'model_calls': 0}))
