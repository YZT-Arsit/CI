"""Freeze replay code during generation, before any output scoring."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
assert not (R / 'SCORING_PLAN.json').exists() and not (R / 'RESULTS.json').exists()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
plan = {'auditor_sha256': sha(R / 'audit.py'),
        'prior_train_plan_sha256': sha(R.parent / '20261117_teacher_binding/PLAN.json'),
        'scorer_frozen_after_worker_launch': True, 'outputs_scored_so_far': False,
        'metrics_predeclared_in': 'PLAN.md', 'whole_goal_complete': False}
(R / 'SCORING_PLAN.json').write_text(json.dumps(plan, indent=2))
print(json.dumps(plan))
