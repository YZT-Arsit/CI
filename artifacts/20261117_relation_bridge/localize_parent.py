"""Post-declared all-state error localization; zero model calls."""
import collections
import json
from dataclasses import asdict
from bridge_data import R, PARENT, prior, read, rows, sha, targets, environment


def main():
    prior.verify(); seal = read(PARENT / 'SEALED.json')
    assert seal['complete'] and all(sha(PARENT / n) == h for n, h in seal['files'].items())
    ws = read(PARENT / 'dev_worlds.json'); plan = read(PARENT / 'PLAN.json')
    summaries = []; failures = []; decisions = episodes = 0
    for arm in plan['arms']:
        counter = collections.Counter()
        for condition, group in ws.items():
            for world, row in zip(group, rows(PARENT / 'study' / arm / (condition + '.jsonl'))):
                env = environment(world); state = env.initial(); cases = targets(world); first_loss = None
                for e in row['trace']:
                    obs = env.observation(state); case = next(c for c in cases if c['observation'] == obs)
                    expected = {c['action']: c['safe_completion'] for c in case['candidates']}
                    query_ok = all(q['estimate'] == str(expected[q['action']]) for q in e['queries'])
                    optimal = e['selected'] in [case['candidates'][i]['action'] for i in case['optimal_indices']]
                    counter[str((e['role'], query_ok, optimal))] += 1; decisions += 1
                    if e['selected'] is None: break
                    if any(expected.values()) and not expected[e['selected']] and first_loss is None:
                        first_loss = {'phase': e['phase'], 'role': e['role'], 'action': e['selected']}
                    after, audit = env.transition(state, e['selected'])
                    assert json.loads(json.dumps(audit)) == e['audit']
                    if after.violation and not state.violation:
                        failures.append({'arm': arm, 'condition': condition, 'world_id': world['world_id'],
                            'role': e['role'], 'feasible_before_violation': any(expected.values()), 'selected': e['selected'],
                            'predictions': {q['action']: q['estimate'] for q in e['queries']}, 'truth': expected,
                            'all_queries_correct': query_ok, 'optimal': optimal, 'first_completion_loss': first_loss})
                    state = after
                assert json.loads(json.dumps(asdict(state))) == row['final']; episodes += 1
        summaries.append({'arm': arm, 'role_query_action_counts': dict(counter)})
    native = read(PARENT / 'POLICY_RESULTS.json'); assert decisions == native['new_decisions'] and episodes == 288
    result = {'summaries': summaries, 'violations': failures, 'decisions': decisions, 'episodes_replayed': episodes,
        'new_model_calls': 0, 'parent_seal_sha256': sha(PARENT / 'SEALED.json'), 'source_sha256': sha(R / 'localize_parent.py'),
        'scope': 'All observed on-policy states, not matched across models; descriptive co-occurrence, not proof that correcting a judgment repairs a trajectory.'}
    (R / 'PARENT_LOCALIZATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'episodes': episodes, 'decisions': decisions, 'global_violations': [f for f in failures if f['arm'].startswith('global')]}))


if __name__ == '__main__': main()
