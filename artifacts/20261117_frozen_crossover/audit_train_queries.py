"""Audit exact TRAIN query prompts after sealing their free responses."""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path[:0] = [str(R.parents[1] / 'code'), str(R.parent / '20261117_binding_factors')]
from render import query


def main():
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    spec = json.loads((R / 'TRAIN_QUERY_PLAN.json').read_text())
    assert sha(R / 'PLAN.json') == spec['parent_plan_sha256']
    parent = json.loads((R / 'PLAN.json').read_text())
    assert all(sha(R.parents[1] / name) == digest for name, digest in parent['dependency_hashes'].items())
    for arm in spec['arms']:
        item = parent['checkpoints'][arm]
        path = R.parents[1] / item['path']
        assert json.loads((path / 'CHECKPOINT_SEALED.json').read_text()) == item['seal']
        assert all(sha(path / 'adapter' / name) == digest for name, digest in item['seal']['adapter_hashes'].items())
    assert all(sha(R / name) == digest for name, digest in spec['source_hashes'].items())
    run = json.loads((R / 'TRAIN_QUERY_RUN_PLAN.json').read_text())
    assert run['plan_sha256'] == sha(R / 'TRAIN_QUERY_PLAN.json') and not run['read_labels']
    seal = json.loads((R / 'TRAIN_QUERIES_SEALED.json').read_text())
    assert seal['complete'] and not seal['labels_scored'] and not seal['read_labels']
    assert all(sha(R / name) == digest for name, digest in seal['files'].items())
    assert sha(R / 'train_oracle.jsonl') == spec['labels_sha256']
    assert sha(R / 'train_observations.jsonl') == spec['observations_sha256']
    labels = list(map(json.loads, (R / 'train_oracle.jsonl').read_text().splitlines()))
    cases = list(map(json.loads, (R / 'train_observations.jsonl').read_text().splitlines()))
    assert cases == [{key: row[key] for key in ['case_id', 'step', 'seeds', 'observation']} for row in labels]
    pool = json.loads((R.parent / '20261116_relation_solution/entity_palettes.json').read_text())['train']
    stats = defaultdict(Counter); seconds = []
    for arm in spec['arms']:
        records = list(map(json.loads, (R / f'train_queries_{arm}.jsonl').read_text().splitlines()))
        assert len(records) == len(cases) == 320
        for start in range(0, len(records), 4):
            assert len({q['seconds_batch'] for row in records[start:start+4] for q in row['queries']}) == 1
        for index, (case, label, record) in enumerate(zip(cases, labels, records)):
            assert record['case_id'] == case['case_id']
            assert [q['action'] for q in record['queries']] == case['observation']['actions']
            truth = {c['action']: str(c['safe_completion']) for c in label['candidates']}
            correct = True
            counter = stats[arm, case['observation']['role']]
            for q in record['queries']:
                assert q['prompt'] == query(case['observation'], pool, case['seeds'], q['action'])
                found = re.findall(r'ANSWER:\s*([01])\b', q['raw'])
                valid = len(found) == 1 and q['seconds_batch'] < 9.5
                bit = found[0] if valid else 'unknown'
                assert q['valid'] == valid and q['estimate'] == bit
                correct &= bit == truth[q['action']]
                counter.update(queries=1, correct=int(bit == truth[q['action']]), invalid=int(not valid))
                counter['truth_' + truth[q['action']]] += 1
                counter['correct_truth_' + truth[q['action']]] += bit == truth[q['action']]
            counter.update(presentations=1, all_correct=int(correct))
            if index % 4 == 0:
                seconds.append(record['queries'][0]['seconds_batch'])
    result = {'metrics': [{'arm': arm, 'role': role, **dict(value)} for (arm, role), value in stats.items()],
              'query_responses': sum(value['queries'] for value in stats.values()), 'generation_batches': len(seconds),
              'generation_seconds_sum': sum(seconds), 'max_generation_seconds': max(seconds),
              'errors': 0, 'new_validation_cases': 0, 'model_updates': 0, 'whole_goal_complete': False}
    assert result['query_responses'] == 1600 and result['generation_batches'] == 160
    (R / 'TRAIN_QUERY_RESULTS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
