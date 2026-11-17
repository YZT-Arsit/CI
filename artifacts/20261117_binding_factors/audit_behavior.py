"""Describe refusal and unsafe failure separately after native replay succeeds."""
import hashlib
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parent / 'repair'


def main():
    assert json.loads((R / 'AUDIT.json').read_text())['errors'] == 0
    all_summaries = json.loads((R / 'RESULTS.json').read_text())['summaries']
    output = {}
    for arm in ['two_view_sft', 'conditional_js']:
        training = [json.loads(x) for x in (R / arm / 'training.jsonl').read_text().splitlines()]
        summaries = [s for s in all_summaries if s['arm'] == arm]
        hist = Counter()
        safe_failed = total = 0
        for summary in summaries:
            path = R / arm / (summary['condition'] + '.jsonl')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == summary['raw_sha256']
            for row in map(json.loads, path.read_text().splitlines()):
                total += 1
                safe_failed += row['ci'] and not row['strict_success']
                for t in row['trace']:
                    action = t['selected'] or 'INVALID'
                    hist[t['role'] + ':' + ('COMMIT' if action.startswith('COMMIT:') else action)] += 1
        output[arm] = {'updates': len(training),
                       'max_update_seconds': max(t['seconds'] for t in training),
                       'input_tokens': sum(t['input_tokens'] for t in training),
                       'actor_consistency_eligible_states': sum(t['actor_consistency_eligible_states'] for t in training),
                       'eligible_state_opportunities': 2 * len(training),
                       'episodes': total, 'safe_failed_episodes': safe_failed,
                       'selected_actions': dict(hist),
                       'totals': {k: sum(s[k] for s in summaries)
                                  for k in ['strict_success', 'valid_ci', 'invalid', 'invalid_queries']}}
    result = {'arms': output,
              'scope': 'Descriptive on-policy counts; condition mixtures and reached states differ. No additional independent samples or conditional causal comparison.'}
    (R / 'BEHAVIOR_SUMMARY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
