"""Freeze original TRAIN anchors and all dependencies before optimizer diagnosis."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R / 'PLAN.json').exists()
old = json.loads((R.parent / '20261117_teacher_binding/PLAN.json').read_text())
cross = json.loads((R.parent / '20261117_frozen_crossover/PLAN.json').read_text())
groups = {}
for i, batch in enumerate(old['batches']):
    groups.setdefault(tuple(batch['key']), []).append(i)
cases = []
for key, indices in groups.items():
    anchor = old['batches'][indices[0]]
    companion = json.loads(json.dumps(old['batches'][indices[-1]]))
    if len(indices) == 1:
        companion['views'] = [{k: v + 1000003 for k, v in view.items()} for view in companion['views']]
    cases.append({'key': list(key), 'anchor_step': indices[0], 'companion_step': indices[-1],
                  'anchor': anchor, 'companion': companion, 'new_companion_view': len(indices) == 1})
deps = dict(cross['dependency_hashes'])
for name in ['20261117_teacher_binding/binding_loss.py', '20261117_sampler_repair/sampling.py',
             '20261117_binding_factors/two_view_loss.py', '20261117_action_binding/coverage_repair.py',
             '20261117_action_binding/probe_frozen.py', '20261117_frozen_crossover/common.py']:
    deps['artifacts/' + name] = sha(R.parent / name)
for name, digest in old['training_files'].items():
    deps['artifacts/20261116_relation_solution/dataset/train/' + name] = digest
assert all(sha(ROOT / name) == digest for name, digest in deps.items())
plan = {'cases': cases, 'arms': ['none', 'query', 'actor', 'joint'], 'seed': 17, 'lr': .0008,
        'max_operation_seconds': 15, 'checkpoint': cross['checkpoints']['teacher_sft'],
        'old_plan_sha256': sha(R.parent / '20261117_teacher_binding/PLAN.json'),
        'cross_plan_sha256': sha(R.parent / '20261117_frozen_crossover/PLAN.json'),
        'dependencies': deps, 'sources': {p.name: sha(p) for p in R.glob('*.py')},
        'no_persistent_model_update': True, 'optimizer_state': 'fresh AdamW for every branch',
        'whole_goal_complete': False}
(R / 'PLAN.json').write_text(json.dumps(plan, indent=2))
print(json.dumps({'cases': len(cases), 'branches': len(cases)*4, 'dependencies': len(deps)}))
