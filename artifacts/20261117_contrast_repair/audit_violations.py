"""Post-result descriptive trace audit; no additional model calls or causal intervention."""
import collections
import json
from paths import R, read, rows, sha
from train_probe import verify


def main():
    verify()
    results = read(R / 'RESULTS.json')
    worlds = read(R / 'dev_worlds.json')
    summaries, details = [], []
    for arm in ['one_hop', 'mixed_depth']:
        counts = collections.Counter()
        for condition in worlds:
            for index, record in enumerate(rows(R / 'repair' / arm / (condition + '.jsonl'))):
                violations = [e for e in record['trace'] if e['audit']['verdict'] == 'VIOLATION']
                assert bool(violations) == (not record['ci'])
                if not violations:
                    continue
                event = violations[0]
                selected = [q for q in event['queries'] if q['action'] == event['selected']]
                assert len(selected) == 1
                query = selected[0]
                estimate = query['estimate'] if query['valid'] else 'invalid'
                counts[event['role'], estimate] += 1
                details.append({'arm': arm, 'condition': condition, 'episode_index': index,
                                'world_id': record['world_id'], 'role': event['role'],
                                'phase': event['phase'], 'selected': event['selected'],
                                'selected_query_estimate': estimate,
                                'violation_events': len(violations)})
        total = sum(counts.values())
        assert total == next(a['violations'] for a in results['arms'] if a['arm'] == arm)
        summaries.append({'arm': arm, 'violating_episodes': total,
                          'first_violation_by_role_estimate': [
                              {'role': role, 'selected_query_estimate': estimate, 'episodes': n}
                              for (role, estimate), n in sorted(counts.items())]})
    output = {'summaries': summaries, 'details': details, 'model_calls': 0, 'errors': 0,
              'main_seal_sha256': sha(R / 'SEALED.json'),
              'results_sha256': sha(R / 'RESULTS.json'),
              'scope': 'Post-result descriptive audit of already replay-verified native traces. A positive estimate followed by a violating action is an observed false positive, not a new counterfactual intervention or proof that replacing it would repair the episode.',
              'primary_gate_changed': False}
    (R / 'VIOLATION_DIAGNOSTIC.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k != 'details'}))


if __name__ == '__main__':
    main()
