"""Finite training-corpus audit of completion bits versus safety/value labels."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent
DATA = R.parent / '20261116_relation_solution/dataset/train'


def main():
    candidate_groups = defaultdict(list)
    state_groups = defaultdict(list)
    joint_groups = defaultdict(set)
    sources = {}
    states = 0
    candidates = 0
    for path in sorted(DATA.glob('states_*.jsonl')):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            obs = row['observation']
            tickets = {p['ticket']: str(i) for i, p in enumerate(obs['contract']['providers'])}
            def normalize(action):
                return 'COMMIT:' + tickets[action.split(':', 1)[1]] if action.startswith('COMMIT:') else action
            actions = tuple(normalize(c['action']) for c in row['candidates'])
            bits = tuple(c['safe_completion'] for c in row['candidates'])
            optimal = {actions[i] for i in row['optimal_indices']}
            state_groups[(obs['role'], actions, bits)].append((row['state_id'], optimal))
            for action, c in zip(actions, row['candidates']):
                key = (obs['role'], action, c['safe_completion'])
                candidate_groups[key].append((row['state_id'], c['immediate_safe'], tuple(c['value'])))
                joint_groups[(obs['role'], action, c['immediate_safe'], c['safe_completion'])].add(tuple(c['value']))
                candidates += 1
            states += 1
    ambiguity = []
    for key, rows in sorted(candidate_groups.items()):
        values = {(safe, value) for _, safe, value in rows}
        if len(values) > 1:
            ambiguity.append({'key': key, 'records': len(rows), 'safety_values': sorted(values),
                              'witnesses': [next({'state_id': sid, 'immediate_safe': s, 'value': v}
                                                for sid, s, v in rows if (s, v) == value)
                                            for value in sorted(values)]})
    conflicts = []
    for key, rows in sorted(state_groups.items()):
        common = set.intersection(*(optimal for _, optimal in rows))
        if not common:
            conflicts.append({'key': key, 'states': len(rows), 'examples': [(s, sorted(o)) for s, o in rows[:4]]})
    result = {'states': states, 'candidates': candidates, 'source_hashes': sources,
              'completion_bit_candidate_groups': len(candidate_groups),
              'candidate_safety_or_value_ambiguities': ambiguity,
              'joint_safety_completion_value_ambiguities': [list(k) for k, v in joint_groups.items() if len(v) > 1],
              'completion_vector_state_groups': len(state_groups),
              'groups_without_common_optimal_action': conflicts,
              'scope': 'Finite existing TRAIN audit; role and provider-index-normalized action identities retained. No model inference, no source generalization proof, no whole-observation information-loss claim.'}
    (R / 'VALUE_CHANNEL_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
