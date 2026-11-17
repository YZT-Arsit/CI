"""Freeze matched observations, checkpoints and exact TRAIN oracle presentations."""
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(R.parent / '20261116_relation_solution'), str(R.parent / '20261117_sampler_repair')]
from experiments.relation_solution_data import targets
from audit_data import check
from sampling import buckets, DATA


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (R / 'PLAN.json').exists()
    prior = R.parent / '20261117_teacher_binding'
    specification = json.loads((prior / 'PLAN.json').read_text())
    worlds_path = prior / 'repair/dev_worlds.json'
    assert sha(worlds_path) == specification['dev_sha256']
    observations, labels = [], []
    for condition, worlds in json.loads(worlds_path.read_text()).items():
        for world in worlds:
            for target in targets(world):
                assert not check(target)
                cid = hashlib.sha256((condition + ':' + target['state_id']).encode()).hexdigest()
                observations.append({'case_id': cid, 'condition': condition, **{key: target[key] for key in
                    ['state_id', 'world_id', 'base_group_id', 'observation']}})
                labels.append({'case_id': cid, **target})
    assert len(observations) == len({row['case_id'] for row in observations}) == 560
    data = buckets()
    train = []
    for name, digest in specification['training_files'].items():
        assert sha(DATA / name) == digest
    for step, batch in enumerate(specification['batches']):
        pair = data[tuple(batch['key'])][batch['index']]
        for view, seeds in enumerate(batch['views']):
            for side, case in enumerate(pair):
                assert not check(case)
                train.append({'case_id': f'train_{step}_{view}_{side}', 'step': step, 'seeds': seeds, **case})
    assert len(train) == 320
    for name, data in [('observations.jsonl', observations), ('labels.jsonl', labels), ('train_oracle.jsonl', train)]:
        (R / name).write_text(''.join(json.dumps(row) + '\n' for row in data), encoding='utf-8')
    arms = ['own_sft', 'teacher_sft', 'earlier_sft']
    dependencies = ['code/kci/relation_language.py', 'code/experiments/relation_solution_data.py',
        'artifacts/20261116_relation_solution/lm_common.py', 'artifacts/20261116_relation_solution/entity_palettes.json',
        'artifacts/20261116_relation_solution/environment.json', 'artifacts/20261117_action_binding/learned_evidence.py',
        'artifacts/20261117_action_binding/query_pilot.py', 'artifacts/20261117_binding_factors/render.py']
    plan = {'arms': arms, 'train_arms': arms[:2], 'batch_states': 4, 'model_updates': 0,
        'state_versions': 560, 'observation_states': 280, 'train_presentations_per_arm': 320,
        'learned_actor_responses': 5040, 'oracle_dev_responses': 1680, 'oracle_train_responses': 640,
        'source_dev_path': worlds_path.relative_to(ROOT).as_posix(), 'source_dev_sha256': sha(worlds_path),
        'checkpoints': {arm: {'path': (prior / 'repair' / arm).relative_to(ROOT).as_posix(),
            'seal': json.loads((prior / 'repair' / arm / 'CHECKPOINT_SEALED.json').read_text())} for arm in arms},
        'source_hashes': {p.name: sha(p) for p in R.glob('*.py')},
        'dependency_hashes': {name: sha(ROOT / name) for name in dependencies},
        'data_hashes': {name: sha(R / name) for name in ['observations.jsonl', 'labels.jsonl', 'train_oracle.jsonl']},
        'same_DEV_reused': True, 'source_holdout': False, 'scope': 'Frozen cross-checkpoint diagnosis; diagonal shared policy, off-diagonal intervention, oracle separate; not a substitute for shared-model research goal.',
        'whole_goal_complete': False}
    (R / 'PLAN.json').write_text(json.dumps(plan, indent=2))
    print(json.dumps({key: plan[key] for key in ['state_versions', 'learned_actor_responses', 'oracle_dev_responses', 'oracle_train_responses']}))


if __name__ == '__main__':
    main()
