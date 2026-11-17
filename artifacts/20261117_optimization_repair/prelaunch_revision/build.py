"""Freeze matched continuation budgets and genuinely new generated DEV."""
import hashlib
import json
from pathlib import Path
from worlds import dev_worlds, targets, check

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R / 'PLAN.json').exists()
old = json.loads((R.parent / '20261117_teacher_binding/PLAN.json').read_text())
probe = json.loads((R.parent / '20261117_optimization_probe/PLAN.json').read_text())
cross = json.loads((R.parent / '20261117_frozen_crossover/PLAN.json').read_text())
worlds, paired = dev_worlds()
previous = json.loads((R.parent / '20261117_teacher_binding/dev_worlds.json').read_text())
assert not {w['world_id'] for ws in worlds.values() for w in ws} & {w['world_id'] for ws in previous.values() for w in ws}
for ws in worlds.values():
    for w in ws:
        assert not [e for row in targets(w) for e in check(row)]
(R / 'dev_worlds.json').write_text(json.dumps(worlds), encoding='utf-8')
deps = dict(probe['dependencies'])
for name in ['20261117_teacher_binding/replay.py', '20261117_frozen_crossover/train_oracle.jsonl']:
    deps['artifacts/'+name] = sha(R.parent / name)
arms = ['continue_original_lr', 'continue_lower_lr', 'teacher_sft', 'earlier_sft']
plan = {'arms': arms, 'evidence_arms': arms, 'batches': old['batches'], 'seed': 17,
        'rates': {'continue_original_lr': .0008, 'continue_lower_lr': .0002},
        'new_updates_per_arm': 80, 'new_total_updates': 160, 'paired_conditions': paired,
        'checkpoints': cross['checkpoints'], 'cross_plan_sha256': sha(R.parent / '20261117_frozen_crossover/PLAN.json'),
        'dependencies': deps, 'source_hashes': {p.name: sha(p) for p in R.glob('*.py')},
        'source_sha256': sha(R / 'run.py'), 'dev_sha256': sha(R / 'dev_worlds.json'),
        'whole_goal_complete': False, 'oracle_at_runtime': False}
(R / 'PLAN.json').write_text(json.dumps(plan, indent=2))
print(json.dumps({'world_presentations': sum(map(len, worlds.values())), 'new_updates':160}))
