"""Post-declared descriptive analysis; never selects checkpoints or alters gates."""
import collections
import json
import statistics
from join_data import R, read, rows, sha, verify


def classify(label, raw):
    # The same binary parser and time cutoff used by the frozen factor scorer.
    from scoring_factors import parse
    value = parse(raw, 'binary')
    return 'invalid' if value is None else 'correct' if value == label else 'false_negative' if label else 'false_positive'


def main():
    import sys
    sys.path.insert(0, str(R.parent / '20261117_relation_factors'))
    plan = verify(); seal = read(R / 'SEALED.json'); assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(R / name) == digest for name, digest in seal['files'].items())
    result = read(R / 'RESULTS.json'); summaries = {s['arm']: s for s in result['summaries']}
    assert len(result['summaries']) == len(summaries) == 9 and set(summaries) == set(plan['arms'])
    assert all(s['seed'] == plan['arms'][arm]['seed'] and s['kind'] == plan['arms'][arm]['kind'] for arm, s in summaries.items())
    categories = {}; transitions = []; losses = []
    for panel in ['standard', 'factors']:
        obs, labels = rows(R / f'{panel}_observations.jsonl'), rows(R / f'{panel}_labels.jsonl')
        for arm, summary in summaries.items():
            raw = rows(R / 'study' / arm / f'{panel}.jsonl'); statuses = []; counts = collections.Counter()
            assert len(raw) == len(obs) == len(labels)
            for o, l, response in zip(obs, labels, raw):
                assert o['id'] == l['id'] == response['id']
                status = classify(l['target'], response); statuses.append(status); counts[status] += 1
            assert counts['correct'] == summary[panel]['correct'] and counts['invalid'] == summary[panel]['invalid']
            categories[arm, panel] = statuses
            summary[panel]['error_categories'] = dict(counts)
        for seed in [17, 29, 41]:
            for left, right in [('initial', 'random'), ('initial', 'targeted'), ('random', 'targeted')]:
                a, b = left + str(seed), right + str(seed); counts = collections.Counter(); slices = collections.defaultdict(collections.Counter)
                for o, l, before, after in zip(obs, labels, categories[a, panel], categories[b, panel]):
                    transition = before + '->' + after; counts[transition] += 1
                    key = str(o.get('kind', o.get('family'))) + '/' + str(o['noise']) + '/label' + str(l['target'])
                    slices[key][transition] += 1
                assert sum(counts.values()) == len(obs)
                transitions.append({'panel': panel, 'from': a, 'to': b, 'counts': dict(counts), 'slices': {k: dict(v) for k, v in slices.items()}})
    for arm, summary in summaries.items():
        if summary['kind'] == 'initial': continue
        records = rows(R / 'study' / arm / 'training.jsonl'); assert len(records) == 96
        fields = ['bit_loss', 'format_loss', 'actor_bit_loss', 'actor_format_loss', 'gradient_norm', 'seconds']
        losses.append({'arm': arm, 'first_24_mean': {k: statistics.mean(v[k] for v in records[:24]) for k in fields},
                       'last_24_mean': {k: statistics.mean(v[k] for v in records[-24:]) for k in fields},
                       'gradient_norm_max': max(v['gradient_norm'] for v in records)})
    output = {'error_categories': {arm: {p: s[p]['error_categories'] for p in ['standard', 'factors']} for arm, s in summaries.items()},
              'matched_transitions': transitions, 'training_descriptives': losses, 'new_model_calls': 0,
              'scope': 'Post-declared descriptive breakdown only; no new gate, tuning or checkpoint selection. Loss trends are not evidence of internal circuits or gradient conflict.',
              'results_sha256': sha(R / 'RESULTS.json'), 'script_sha256': sha(R / 'diagnostics.py')}
    (R / 'DIAGNOSTICS.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'matched_comparisons': len(transitions), 'model_panels': len(categories), 'new_model_calls': 0}))


if __name__ == '__main__': main()
