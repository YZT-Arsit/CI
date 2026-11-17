"""Summarize sealed, natively audited rollout behavior without further inference."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parent


def main():
    audit = json.loads((R / 'repair/AUDIT.json').read_text())
    assert audit['errors'] == 0
    result = {}
    for arm in ['sampler_sft', 'full_js', 'query_js', 'value_js']:
        root = R / 'repair' / arm
        episodes = [json.loads(line) for path in sorted(root.glob('*.jsonl'))
                    if path.name != 'training.jsonl' for line in path.read_text().splitlines()]
        assert len(episodes) == 112
        decisions = [step for row in episodes for step in row['trace']]
        base = [json.loads(line) for path in sorted(root.glob('base_*.jsonl'))
                for line in path.read_text().splitlines()]
        counts = {}
        for role in ['Planner', 'Source', 'Worker']:
            counts[role] = dict(Counter(step['selected'] if role != 'Planner' else step['raw'].strip()
                                       for row in base for step in row['trace'] if step['role'] == role))
        training = root / 'training.jsonl'
        updates = [json.loads(line) for line in training.read_text().splitlines()] if training.exists() else []
        result[arm] = {
            'episodes': len(episodes), 'strict_success': sum(row['strict_success'] for row in episodes),
            'valid_ci': sum(row['ci'] and not row['invalid'] for row in episodes),
            'invalid_episodes': sum(row['invalid'] for row in episodes),
            'invalid_queries': sum(row['invalid_queries'] for row in episodes),
            'base_decision_counts_on_reached_states': counts,
            'new_updates': len(updates),
            'max_update_seconds': max((row['seconds'] for row in updates), default=None),
            'max_generation_seconds': max([step['seconds_batch'] for step in decisions] +
                                          [q['seconds_batch'] for step in decisions for q in step['queries']]),
        }
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
