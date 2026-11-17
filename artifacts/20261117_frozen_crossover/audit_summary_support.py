"""Label-only summary sufficiency check; no model outputs or model calls."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent


def name(row, action):
    if action.startswith('COMMIT:'):
        return 'COMMIT:' + str(next(i for i, p in enumerate(row['observation']['contract']['providers'])
                                   if action == 'COMMIT:' + p['ticket']))
    return action


def group(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        candidates = sorted(row['candidates'], key=lambda c: name(row, c['action']))
        key = (row['observation']['role'], tuple((name(row, c['action']), tuple(c[f] for f in fields)) for c in candidates))
        best = {name(row, row['candidates'][i]['action']) for i in row['optimal_indices']}
        groups[key].append((row, best))
    collisions = []
    for key, members in groups.items():
        common = set.intersection(*(best for _, best in members))
        if not common:
            remaining = set.union(*(best for _, best in members))
            witnesses = []
            for row, best in members:
                reduced = remaining & best
                if reduced != remaining:
                    witnesses.append({'case_id': row['case_id'], 'state_id': row['state_id'], 'optimal': sorted(best)})
                    remaining = reduced
                if not remaining:
                    break
            assert not set.intersection(*(set(row['optimal']) for row in witnesses))
            collisions.append({'role': key[0], 'summary': key[1], 'presentations': len(members), 'witnesses': witnesses})
    return {'summary_groups': len(groups), 'incompatible_groups': len(collisions), 'collisions': collisions}


def main():
    plan = json.loads((R / 'PLAN.json').read_text())
    datasets = {}
    for split, filename in [('train', 'train_oracle.jsonl'), ('dev', 'labels.jsonl')]:
        data = (R / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == plan['data_hashes'][filename]
        datasets[split] = list(map(json.loads, data.decode('utf-8').splitlines()))
    datasets['train_and_dev'] = datasets['train'] + datasets['dev']
    results = {split: {'presentations': len(rows), 'completion_only': group(rows, ['safe_completion']),
                      'safety_and_completion': group(rows, ['immediate_safe', 'safe_completion'])}
               for split, rows in datasets.items()}
    constants = defaultdict(Counter)
    for row in datasets['dev']:
        counter = constants[row['observation']['role']]
        counter['states'] += 1
        for i in row['optimal_indices']:
            counter[name(row, row['candidates'][i]['action'])] += 1
    output = {'results': results, 'model_calls': 0, 'model_outputs_read': False,
              'dev_constant_action_optimal_counts': {role: dict(v) for role, v in constants.items()},
              'constant_action_scope': 'Finite label diagnostic including ties and repeated views; COMMIT normalized by provider index, not a learned policy.',
              'scope': 'Finite label-only summary compatibility; actor also sees full observations. Not proof of learnability or natural-norm validity.',
              'whole_goal_complete': False}
    (R / 'SUMMARY_SUPPORT.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output))


if __name__ == '__main__':
    main()
