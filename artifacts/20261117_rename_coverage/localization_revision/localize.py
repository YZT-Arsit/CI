"""Descriptive failure localization; never a causal intervention or training signal."""
import collections
import json
from rename_data import R, read, rows, sha, verify, targets
from experiments.relation_solution_data import environment


def main():
    plan = verify(); seal = read(R / 'SEALED.json')
    assert all(sha(R / n) == h for n, h in seal['files'].items())
    registration = read(R / 'LOCALIZATION_PLAN.json')
    assert registration['source_sha256'] == sha(R / 'localize.py') and registration['plan_sha256'] == sha(R / 'PLAN.json')
    worlds = read(R / 'dev_worlds.json'); summaries = []; detailed = []
    for arm in plan['arms']:
        counters = collections.Counter(); first_errors = collections.Counter()
        for condition, ws in worlds.items():
            results = rows(R / 'study' / arm / (condition + '.jsonl'))
            for world, result in zip(ws, results):
                assert world['world_id'] == result['world_id']; cases = targets(world)
                solvable = any(c['safe_completion'] for c in cases[0]['candidates'])
                counters['solvable' if solvable else 'unsolvable'] += 1
                counters['solvable_success'] += int(solvable and result['strict_success'])
                counters['unsolvable_valid_safe_stop'] += int(not solvable and result['ci'] and not result['invalid'] and not result['invalid_queries'] and result['final']['done'])
                env = environment(world); state = env.initial(); first = None
                for entry in result['trace']:
                    observation = env.observation(state); case = next(c for c in cases if c['observation'] == observation)
                    best = {case['candidates'][i]['action'] for i in case['optimal_indices']}
                    query_correct = all(q['valid'] and q['estimate'] == str(next(c['safe_completion'] for c in case['candidates'] if c['action'] == q['action'])) for q in entry['queries'])
                    if first is None and entry['selected'] not in best:
                        first = {'role': entry['role'], 'phase': entry['phase'], 'query_vector_correct': query_correct,
                                 'selected': entry['selected'], 'selected_estimate': next((q['estimate'] for q in entry['queries'] if q['action'] == entry['selected']), None)}
                    if entry['selected'] is None: break
                    state, audit = env.transition(state, entry['selected'])
                if not result['strict_success']:
                    if first is None: first_errors['no_suboptimal_action'] += 1
                    else: first_errors[first['role'] + ('_queries_correct' if first['query_vector_correct'] else '_queries_incorrect')] += 1
                detailed.append({'arm': arm, 'condition': condition, 'world_id': world['world_id'], 'solvable': solvable,
                                 'strict_success': result['strict_success'], 'first_suboptimal_action': first})
        assert counters['solvable'] == 120 and counters['unsolvable'] == 8
        summaries.append({'arm': arm, **dict(counters), 'failed_episode_first_suboptimal': dict(first_errors)})
    result = {'summaries': summaries, 'episodes': detailed, 'model_calls': 0,
              'scope': 'Descriptive localization on model-visited states; no causal attribution, no new success gate.', 'whole_goal_complete': False}
    (R / 'LOCALIZATION_RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'summaries': summaries, 'model_calls': 0}))


if __name__ == '__main__':
    main()
