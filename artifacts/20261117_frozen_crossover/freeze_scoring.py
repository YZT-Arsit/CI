"""Freeze replay code during generation, before any output scoring."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
assert not (R / 'SCORING_PLAN.json').exists() and not (R / 'RESULTS.json').exists()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
spec = json.loads((R / 'PLAN.json').read_text())
prior_results = R.parent / '20261117_teacher_binding/repair/RESULTS.json'
counts = {arm: 0 for arm in spec['arms']}
for report in json.loads(prior_results.read_text())['summaries']:
    if report['arm'] in counts:
        path = R.parents[1] / spec['checkpoints'][report['arm']]['path'] / (report['condition'] + '.jsonl')
        assert sha(path) == report['raw_sha256']
        values = list(map(json.loads, path.read_text().splitlines()))
        assert len(values) == report['episodes']
        counts[report['arm']] += sum(len(row['trace']) for row in values)
plan = {'auditor_sha256': sha(R / 'audit.py'),
        'prior_train_plan_sha256': sha(R.parent / '20261117_teacher_binding/PLAN.json'),
        'prior_results_sha256': sha(prior_results), 'preflight_sha256': sha(R / 'PREFLIGHT.json'),
        'prior_expected_decisions': counts,
        'scorer_frozen_after_worker_launch': True, 'outputs_scored_so_far': False,
        'metrics_predeclared_in': 'PLAN.md', 'whole_goal_complete': False}
(R / 'SCORING_PLAN.json').write_text(json.dumps(plan, indent=2))
print(json.dumps(plan))
