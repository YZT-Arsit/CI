import collections
import json
from join_data import R, read, rows, sha, verify
verify(); results = []; base_ids = set(); variants = 0
for seed in [17, 29, 41]:
    data = read(R / f'worlds_{seed}.json'); schedule = rows(R / f'schedule_{seed}.jsonl'); counts = collections.Counter()
    nodes = {kind: collections.Counter() for kind in data}; incoming = {kind: collections.Counter() for kind in data}
    for key, a in data['random'].items():
        b = data['targeted'][key]; assert key not in base_ids; base_ids.add(key)
        control = a['kind'] != 'switch'; variants += 1 if control else 2
        assert a['frontier'] == b['frontier'] and a['target'] == b['target'] and len(a['edges']) == len(b['edges'])
        for kind, w in [('random', a), ('targeted', b)]:
            matrix = [[int([i, j] in w['edges']) for j in range(w['n'])] for i in range(w['n'])]
            truth = int(w['target'] in w['frontier'] or any(matrix[i][w['target']] for i in w['frontier']))
            counts[kind + '_positive'] += truth
            shown = set(w['frontier']) | {w['target']} | {v for e in w['edges'] for v in e}; nodes[kind][len(shown)] += 1
            incoming[kind][sum(v == w['target'] and u not in w['frontier'] for u, v in w['edges'])] += 1
        counts['control_records' if control else 'switch_records'] += 1
    order = collections.Counter(tuple(e['actions']) for b in schedule for e in b['actor'])
    assert len(order) == 6 and set(order.values()) == {64}
    results.append({'seed': seed, 'counts': dict(counts), 'printed_nodes': {k: dict(v) for k, v in nodes.items()},
                    'unknown_source_incoming_edges': {k: dict(v) for k, v in incoming.items()}, 'each_option_order_presentations': 64})
assert len(base_ids) == 1152 and variants == 2016
result = {'schedules': results, 'base_records': len(base_ids), 'generated_record_variants': variants,
          'independent_business_tasks_claimed': False, 'model_calls': 0, 'plan_sha256': sha(R / 'PLAN.json'), 'script_sha256': sha(R / 'data_audit.py')}
(R / 'DATA_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'base_records': len(base_ids), 'record_variants': variants, 'errors': 0}))
