"""Count actual scheduled action-value coverage, rather than corpus inventory."""
import collections
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_sampler_repair'))
from sampling import buckets, DATA


def main():
    source = R.parent / '20261117_binding_factors/repair/PLAN.json'
    plan = json.loads(source.read_text())
    data = buckets()
    counts = collections.defaultdict(collections.Counter)
    for batch in plan['batches']:
        for row in data[tuple(batch['key'])][batch['index']]:
            c = counts[row['observation']['role']]
            c['state_presentations_before_two_views'] += 1
            c['unique_optimum'] += len(row['optimal_indices']) == 1
            c['all_actions_tied'] += len(row['optimal_indices']) == len(row['candidates'])
            c['different_value_classes'] += len({tuple(x['value']) for x in row['candidates']}) > 1
    result = {'by_role': dict(counts), 'plan_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('states_*.jsonl'))},
              'sampling_source_sha256': hashlib.sha256((R.parent / '20261117_sampler_repair/sampling.py').read_bytes()).hexdigest(),
              'scope': 'Actual frozen80batch schedule. Counts reuse TRAIN states and omit duplicate second views; not independent samples.'}
    (R / 'TRAIN_VALUE_CLASSES.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['by_role']))


if __name__ == '__main__':
    main()
