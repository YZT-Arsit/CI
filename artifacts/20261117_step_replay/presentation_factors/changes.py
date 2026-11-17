"""Descriptive paired response changes; no new inference or gate changes."""
import collections
import json
import re
from factor_data import R, read, rows, sha, check


check(); seal = read(R / 'SEALED.json')
assert seal['complete'] and all(sha(R / n) == h for n, h in seal['files'].items())
obs = {o['id']: o for o in rows(R / 'observations.jsonl')}; labels = {r['id']: r['target'] for r in rows(R / 'labels.jsonl')}
results = []
for path in sorted((R / 'outputs').glob('*.jsonl')):
    cases = collections.defaultdict(dict)
    for r in rows(path):
        o = obs[r['id']]; matches = re.findall(r'ANSWER:\s*([01])\b', r['raw'])
        bit = int(matches[0]) if len(matches) == 1 and r['seconds_batch'] < 9.5 else None
        cases[o['seed'], o['world_id']][o['view']] = bit, labels[r['id']]
    counts = collections.defaultdict(collections.Counter)
    for (seed, _), variants in cases.items():
        assert set(variants) == {'seen', 'names_only', 'order_only', 'both'} and len({v[1] for v in variants.values()}) == 1
        old, target = variants['seen']
        for name in ['names_only', 'order_only', 'both']:
            new = variants[name][0]
            counts[seed, name].update(cases=1, changed=int(new != old), correct_to_wrong=int(old == target and new != target),
                                      wrong_to_correct=int(old != target and new == target), invalid_before=int(old is None), invalid_after=int(new is None))
    for (seed, variant), stats in counts.items():
        assert stats['cases'] == 16
        results.append({'arm': path.stem, 'schedule': seed, 'variant': variant, **stats})
result = {'summaries': results, 'source_seal_sha256': sha(R / 'SEALED.json'), 'script_sha256': sha(R / 'changes.py'),
          'post_scoring_descriptive_analysis': True, 'main_gate_unchanged': True, 'model_calls': 0}
(R / 'CHANGES.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
