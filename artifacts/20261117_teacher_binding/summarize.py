"""Audit teacher-only TRAIN inputs and summarize free-running DEV outcomes."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_sampler_repair'))
from sampling import buckets, DATA
sys.path.insert(0, str(R))
from replay import check, environment, targets


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    plan = read(R / 'PLAN.json')
    source = read(R.parent / '20261117_sampler_repair/PLAN.json')
    assert plan['batches'] == source['batches']
    assert plan['batches'] == read(R.parent / '20261117_value_consistency/PLAN.json')['batches']
    for name, digest in plan['training_files'].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest
    data = buckets()
    training = {}
    checked = set()
    for arm in plan['query_weights']:
        rows = [json.loads(line) for line in (R / 'repair' / arm / 'training.jsonl').read_text().splitlines()]
        assert len(rows) == len(plan['batches']) == 80
        changes, mismatches, totals = Counter(), Counter(), Counter()
        for step, (batch, row) in enumerate(zip(plan['batches'], rows)):
            assert row['step'] == step + 1 and row['role'] == batch['key'][0]
            assert row['seconds'] < 9.5 and row['actor_js'] == 0
            pair = data[tuple(batch['key'])][batch['index']]
            truth = [{c['action']: str(c['safe_completion']) for c in case['candidates']} for case in pair]
            assert row['actor_input_estimates'] == [truth, truth]
            for case in pair:
                key = json.dumps(case, sort_keys=True)
                if key not in checked:
                    assert not check(case)
                    checked.add(key)
            for view in row['predicted_estimates']:
                assert len(view) == 2
                for actual, expected in zip(view, truth):
                    assert actual.keys() == expected.keys()
                    assert all(value in ['0', '1', 'unknown'] for value in actual.values())
                    totals[row['role']] += 1
                    changes[row['role']] += actual != expected
                    mismatches[row['role']] += sum(actual[key] != expected[key] for key in expected)
        training[arm] = {'updates': len(rows), 'actor_input_maps': dict(totals),
                         'maps_different_from_own_predictions': dict(changes),
                         'bits_different_from_own_predictions': dict(mismatches),
                         'max_update_seconds': max(row['seconds'] for row in rows),
                         'all_actor_training_maps_equal_native_labels': True}
    summaries = read(R / 'repair/RESULTS.json')['summaries']
    worlds = read(R / 'repair/dev_worlds.json')
    outcomes = {}
    for arm in plan['arms']:
        reports = [r for r in summaries if r['arm'] == arm]
        counts = {key: sum(row[key] for row in reports) for key in
                  ['episodes', 'strict_success', 'safe_success', 'valid_ci', 'invalid', 'invalid_queries']}
        actions = {}
        violations = 0
        binding = {}
        for report in reports:
            path = R / 'repair' / arm / (report['condition'] + '.jsonl')
            episodes = list(map(json.loads, path.read_text().splitlines()))
            violations += sum(not episode['ci'] for episode in episodes)
            for world, episode in zip(worlds[report['condition']], episodes):
                env = environment(world)
                state = env.initial()
                cases = targets(world)
                for entry in episode['trace']:
                    observation = env.observation(state)
                    case = next(case for case in cases if case['observation'] == observation)
                    truth = {c['action']: str(c['safe_completion']) for c in case['candidates']}
                    actual = {q['action']: q['estimate'] for q in entry['queries']}
                    query_correct = actual == truth
                    optimal = {case['candidates'][i]['action'] for i in case['optimal_indices']}
                    actor_correct = entry['selected'] in optimal
                    key = f'query_{query_correct}_actor_{actor_correct}'
                    for scope in ['all', 'base'] if report['condition'].startswith('base_') else ['all']:
                        binding.setdefault(scope, {}).setdefault(entry['role'], Counter())[key] += 1
                    if entry['format_valid']:
                        state, _ = env.transition(state, entry['selected'])
            if report['condition'].startswith('base_'):
                for episode in episodes:
                    for entry in episode['trace']:
                        actions.setdefault(entry['role'], Counter())[str(entry['selected'])] += 1
        counts['base_action_counts'] = {role: dict(counter) for role, counter in actions.items()}
        counts['ci_violations'] = violations
        counts['on_policy_binding_not_matched_across_arms'] = binding
        outcomes[arm] = counts
    audit = read(R / 'repair/AUDIT.json')
    assert audit['episodes_replayed'] == 560 and audit['errors'] == 0
    result = {'training': training, 'native_training_states_checked': len(checked), 'outcomes': outcomes,
              'view_results': read(R / 'repair/VIEW_AUDIT.json')['results'],
              'audit_counts': {key: audit[key] for key in ['episodes_replayed', 'decisions_replayed', 'queries_reconstructed', 'independent_label_states', 'errors']},
              'new_optimizer_updates': 160, 'runtime_teacher_labels': False, 'runtime_action_mask': False,
              'source_holdout': False, 'whole_goal_complete': False}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
