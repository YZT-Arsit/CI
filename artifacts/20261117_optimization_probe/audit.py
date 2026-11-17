"""Verify restored-step experiment records and report local loss effects."""
import hashlib
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    plan = read(R / 'PLAN.json'); run = read(R / 'RUN_PLAN.json'); seal = read(R / 'SEALED.json')
    scoring = read(R / 'SCORING_PLAN.json')
    assert scoring['auditor_sha256'] == sha(Path(__file__))
    assert run['plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(R / name) == digest for name, digest in plan['sources'].items())
    assert all(sha(ROOT / name) == digest for name, digest in plan['dependencies'].items())
    assert sha(R.parent / '20261117_teacher_binding/PLAN.json') == plan['old_plan_sha256']
    assert sha(R.parent / '20261117_frozen_crossover/PLAN.json') == plan['cross_plan_sha256']
    checkpoint = ROOT / plan['checkpoint']['path']
    assert read(checkpoint / 'CHECKPOINT_SEALED.json') == plan['checkpoint']['seal']
    assert all(sha(checkpoint / 'adapter' / name) == digest for name, digest in plan['checkpoint']['seal']['adapter_hashes'].items())
    assert seal['complete'] and seal['restored_parameter_sha256'] == run['initial_parameter_sha256']
    assert all(sha(R / name) == digest for name, digest in seal['files'].items())
    assert not (R / 'TIMEOUT.json').exists()
    records = [json.loads(line) for line in (R / 'records.jsonl').read_text().splitlines()]
    assert len(records) == len(plan['cases']) == 17
    for i, (row, case) in enumerate(zip(records, plan['cases'])):
        assert row['index'] == i and row['key'] == case['key']
        expected = row['dot'] / math.sqrt(row['query_norm2'] * row['actor_norm2'])
        assert math.isclose(row['cosine'], expected, rel_tol=1e-12, abs_tol=1e-12)
        assert [b['arm'] for b in row['branches']] == plan['arms']
        for branch in row['branches']:
            arm = branch['arm']
            norm2 = {'none': 0, 'query': row['query_norm2'], 'actor': row['actor_norm2'],
                     'joint': row['query_norm2']+row['actor_norm2']+2*row['dot']}[arm]
            assert math.isclose(branch['gradient_norm_before_clip']**2, norm2, rel_tol=1e-5, abs_tol=1e-7)
            for location in ['anchor', 'companion']:
                for loss in ['query', 'actor']:
                    value = branch[location+'_after'][loss]
                    assert math.isfinite(value) and value >= 0
                    if arm == 'none':
                        assert abs(value-row[location+'_before'][loss]) < 1e-6
            if arm == 'none':
                assert branch['parameter_sha256'] == run['initial_parameter_sha256']
    timings = read(R / 'TIMINGS.json')
    expected_names = []
    for i in range(17):
        expected_names += [f'{i}/gradients', f'{i}/companion_before']
        expected_names += [f'{i}/{arm}/{location}' for arm in plan['arms'] for location in ['anchor','companion']]
    assert [t['operation'] for t in timings] == expected_names
    assert all(0 <= t['seconds'] < plan['max_operation_seconds'] for t in timings)
    metrics = []
    for arm in plan['arms']:
        for location in ['anchor', 'companion']:
            for loss in ['query', 'actor']:
                changes = [next(b for b in row['branches'] if b['arm']==arm)[location+'_after'][loss]-row[location+'_before'][loss] for row in records]
                metrics.append({'arm': arm, 'location': location, 'loss': loss, 'cases': len(changes),
                    'reduced': sum(x < -1e-5 for x in changes), 'increased': sum(x > 1e-5 for x in changes),
                    'mean_delta': sum(changes)/len(changes), 'min_delta': min(changes), 'max_delta': max(changes)})
    result = {'cases': 17, 'negative_gradient_cosines': sum(row['cosine'] < 0 for row in records),
              'metrics': metrics, 'operations': len(timings), 'max_seconds': max(t['seconds'] for t in timings),
              'operation_seconds_sum': sum(t['seconds'] for t in timings), 'errors': 0,
              'scope': 'Restored one-step fresh-Adam TRAIN diagnostics only; no generated-policy or generalization evidence.',
              'temporary_optimizer_steps': 51, 'persistent_model_updates': 0, 'whole_goal_complete': False,
              'auditor_sha256': sha(Path(__file__))}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
