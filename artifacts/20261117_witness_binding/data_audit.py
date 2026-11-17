"""Describe actual graph coverage, distinguishing node domain from printed nodes."""
import collections
import json
from graph_data import R, read, rows, sha, verify, witnesses, truth
verify()


def summarize(worlds):
    counts = {k: collections.Counter() for k in ['nodes_domain', 'nodes_presented', 'known_nodes', 'edges', 'witness_count', 'target', 'family']}
    for w in worlds:
        shown = set(w['frontier']) | {w['target']} | {v for e in w['edges'] for v in e}
        values = [w['n'], len(shown), len(w['frontier']), len(w['edges']), len(witnesses(w)), truth(w), w['family']]
        for key, value in zip(counts, values): counts[key][str(value)] += 1
    return {'records': len(worlds), 'unique_ids': len({w['id'] for w in worlds}), **{k: dict(v) for k, v in counts.items()}}


train = {str(s): summarize(list(read(R / f'worlds_{s}.json').values())) for s in [17, 29, 41]}
unique = {r['world']['id']: r['world'] for r in rows(R / 'labels.jsonl')}
test = summarize(list(unique.values()))
train_ids = {k for s in [17, 29, 41] for k in read(R / f'worlds_{s}.json')}
assert len(train_ids) == 2304 and not train_ids & set(unique)
random_cells = collections.Counter((w['n'], len(w['edges']), len(w['frontier']), truth(w)) for w in unique.values() if w['kind'] == 'random')
assert len(random_cells) == 16 and set(random_cells.values()) == {2}
result = {'train': train, 'confirmation_unique_worlds': test, 'zero_train_confirmation_id_overlap': True,
          'random_cells': [{'nodes_domain': n, 'edges': m, 'known_nodes': k, 'label': label, 'count': count} for (n, m, k, label), count in random_cells.items()],
          'unprinted_isolated_nodes_are_not_input': True, 'independent_business_tasks_claimed': False,
          'plan_sha256': sha(R / 'PLAN.json'), 'script_sha256': sha(R / 'data_audit.py'), 'model_calls': 0}
(R / 'DATA_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'train_records': len(train_ids), 'confirmation_worlds': len(unique), 'random_cells_balanced': 16, 'errors': 0}))
