"""Audit information lost by completion-only summaries, without model execution."""
import collections
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_sampler_repair'))
from sampling import buckets, DATA


def action_name(row, action):
    if action.startswith('COMMIT:'):
        providers = row['observation']['contract']['providers']
        return 'COMMIT:' + str(next(i for i, p in enumerate(providers) if action == 'COMMIT:' + p['ticket']))
    return action


def grouping(rows, fields):
    groups = collections.defaultdict(list)
    for row in rows:
        candidates = sorted(row['candidates'], key=lambda c: action_name(row, c['action']))
        key = (row['observation']['role'], tuple((action_name(row, c['action']),
                                               tuple(c[field] for field in fields)) for c in candidates))
        optimal = {action_name(row, row['candidates'][i]['action']) for i in row['optimal_indices']}
        groups[key].append((row, optimal))
    collisions = []
    for key, items in groups.items():
        intersection = set.intersection(*(good for _, good in items))
        if not intersection:
            witnesses = []
            for row, good in items:
                if not witnesses or good.isdisjoint(set(witnesses[0]['optimal'])):
                    witnesses.append({'state_id': row['state_id'], 'world_id': row['world_id'],
                                      'optimal': sorted(good), 'candidates': [
                                          {k: c[k] for k in ['action', 'immediate_safe', 'safe_completion', 'value']}
                                          for c in row['candidates']]})
                if len(witnesses) == 2:
                    break
            collisions.append({'role': key[0], 'summary': key[1], 'state_count': len(items),
                               'optimal_action_intersection': [], 'example_rows': witnesses})
    return {'summary_groups': len(groups), 'incompatible_groups': len(collisions), 'collisions': collisions}


def main():
    data = buckets()
    occurrences = [row for pairs in data.values() for pair in pairs for row in pair]
    unique = {}
    for row in occurrences:
        if row['state_id'] in unique:
            previous = unique[row['state_id']]
            assert previous['observation'] == row['observation']
            assert previous['optimal_indices'] == row['optimal_indices']
            assert [(c['action'], c['value'], c['immediate_safe'], c['safe_completion']) for c in previous['candidates']] == [
                (c['action'], c['value'], c['immediate_safe'], c['safe_completion']) for c in row['candidates']]
        unique[row['state_id']] = row
    rows = list(unique.values())
    result = {'training_state_occurrences': len(occurrences), 'unique_observation_states': len(rows), 'completion_only': grouping(rows, ['safe_completion']),
              'safety_and_completion': grouping(rows, ['immediate_safe', 'safe_completion'])}
    log_counts = {}
    for arm in ['repaired_sft', 'repaired_js']:
        source = R.parent / '20261117_sampler_repair'
        plan = json.loads((source / 'PLAN.json').read_text())
        records = list(map(json.loads, (source / 'repair' / arm / 'training.jsonl').read_text().splitlines()))
        assert len(records) == len(plan['batches']) == 80
        counts = collections.defaultdict(collections.Counter)
        for batch, record in zip(plan['batches'], records):
            pair = data[tuple(batch['key'])][batch['index']]
            for estimates in record['predicted_estimates']:
                for row, predicted in zip(pair, estimates):
                    c = counts[row['observation']['role']]
                    truth = {candidate['action']: str(candidate['safe_completion']) for candidate in row['candidates']}
                    assert set(predicted) == set(truth)
                    positive = {a for a, bit in predicted.items() if bit == '1'}
                    optimal = {row['candidates'][i]['action'] for i in row['optimal_indices']}
                    c['presentations'] += 1
                    c['incorrect_map'] += predicted != truth
                    c['predicts_no_completion_when_possible'] += not positive and '1' in truth.values()
                    c['predicted_positive_set_disjoint_from_optimal'] += bool(positive) and not (positive & optimal)
        log_counts[arm] = dict(counts)
    result.update(actual_training_inputs=log_counts,
                  training_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('states_*.jsonl'))},
                  model_calls=0, optimizer_updates=0,
                  scope='TRAIN-only information diagnostic. Current actor also receives full observations, so collisions disprove sufficiency of the completion-summary alone, not identifiability of the full actor input. Two-field sufficiency here is finite-data only, not proof of learnability or transfer.')
    (R / 'CONTRACT_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v if k == 'actual_training_inputs' else {a: b for a, b in v.items() if a != 'collisions'} if isinstance(v, dict) else v
                      for k, v in result.items() if k not in ['training_hashes', 'scope']}))


if __name__ == '__main__':
    main()
