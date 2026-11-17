import importlib.util
import json
import math
from policy_data import R, ROOT, PARENT, read, rows, sha, verify, targets, environment
spec = importlib.util.spec_from_file_location('native_replay_for_join_policy', R / 'native_replay.py')
replay = importlib.util.module_from_spec(spec); spec.loader.exec_module(replay)


def main():
    plan = verify(); seal = read(R / 'SEALED.json'); parent = read(PARENT / 'SEALED.json')
    assert seal['complete'] and parent['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json')
    assert parent['plan_sha256'] == sha(PARENT / 'PLAN.json') and seal['parent_seal_sha256'] == sha(PARENT / 'SEALED.json')
    assert all(sha(R / n) == h for n, h in seal['files'].items()) and all(sha(PARENT / n) == h for n, h in parent['files'].items())
    runtime = read(R / 'study/PLAN.json'); assert runtime['arms'] == list(plan['arms']) and runtime['evidence_arms'] == list(plan['arms'])
    assert all(runtime[k] == v for k, v in plan.items() if k != 'arms') and runtime['parent_seal_sha256'] == sha(PARENT / 'SEALED.json')
    assert (R / 'study/dev_worlds.json').read_bytes() == (R / 'dev_worlds.json').read_bytes()
    parent_plan = read(PARENT / 'PLAN.json')
    for arm, cfg in plan['arms'].items():
        source = PARENT / 'study' / arm; identity = read(R / 'study' / arm / 'EVAL_IDENTITY.json')
        cp = parent_plan['checkpoints'][str(cfg['seed'])] if cfg['kind'] == 'initial' else {'path': source.relative_to(ROOT).as_posix(), 'seal': read(source / 'CHECKPOINT_SEALED.json')}
        assert runtime['checkpoint_bindings'][arm] == identity['checkpoint'] == cp
        assert identity['before'] == identity['after'] == read(source / 'EVAL_IDENTITY.json')['before'] and identity['errors'] == 0
    replay.run('study', 'run_policy.py'); native = read(R / 'study/AUDIT.json')
    assert native['episodes_replayed'] == 288 and native['errors'] == 0
    worlds = read(R / 'dev_worlds.json'); summaries = []; times = []
    for arm, cfg in plan['arms'].items():
        count = dict(episodes=0, strict_success=0, violations=0, invalid=0, invalid_queries=0,
                     feasible_episodes=0, feasible_strict_success=0, infeasible_episodes=0, infeasible_valid_ci=0,
                     decisions=0, optimal_decisions=0, all_actions_optimal_episodes=0,
                     infeasible_decisions=0, infeasible_optimal_decisions=0, infeasible_all_actions_optimal_episodes=0)
        results = {}
        for condition, ws in worlds.items():
            records = rows(R / 'study' / arm / (condition + '.jsonl')); results[condition] = records
            batch_times = {}
            for world, row in zip(ws, records):
                env = environment(world); state = env.initial(); cases = targets(world); obs = env.observation(state)
                initial = next(c for c in cases if c['observation'] == obs); feasible = any(c['safe_completion'] for c in initial['candidates'])
                count['episodes'] += 1; count['strict_success'] += row['strict_success']; count['violations'] += not row['ci']
                count['invalid'] += row['invalid']; count['invalid_queries'] += row['invalid_queries']
                count['feasible_episodes'] += feasible; count['feasible_strict_success'] += feasible and row['strict_success']
                count['infeasible_episodes'] += not feasible; count['infeasible_valid_ci'] += not feasible and row['ci'] and not row['invalid']
                all_optimal = True
                for entry in row['trace']:
                    obs = env.observation(state); case = next(c for c in cases if c['observation'] == obs)
                    optimal = entry['selected'] in [case['candidates'][i]['action'] for i in case['optimal_indices']]
                    all_optimal = all_optimal and optimal; count['decisions'] += 1; count['optimal_decisions'] += optimal
                    count['infeasible_decisions'] += not feasible; count['infeasible_optimal_decisions'] += not feasible and optimal
                    for key, elapsed in [('actor', entry['seconds_batch']), *[('query', q['seconds_batch']) for q in entry['queries']]]:
                        assert math.isfinite(elapsed) and elapsed > 0
                        pair = entry['phase'], key
                        if pair in batch_times: assert batch_times[pair] == elapsed
                        else: batch_times[pair] = elapsed
                    if entry['format_valid']: state, _ = env.transition(state, entry['selected'])
                count['all_actions_optimal_episodes'] += all_optimal
                count['infeasible_all_actions_optimal_episodes'] += not feasible and all_optimal
            times.extend(batch_times.values())
        grouped = {}
        for family in ['base', 'depth3']:
            presentations = [results[family + '_v' + str(v) + suffix] for v in [0, 1] for suffix in ['', '_swap']]
            grouped[family] = {'groups': 2, 'all_four_strict_success': sum(all(p[i]['strict_success'] for p in presentations) for i in range(2))}
        assert count['episodes'] == 32 and count['feasible_episodes'] + count['infeasible_episodes'] == 32
        summaries.append({'arm': arm, 'seed': cfg['seed'], 'kind': cfg['kind'], **count, 'paired_families': grouped})
    calls = native['decisions_replayed'] + native['queries_reconstructed']; assert calls <= 2880
    result = {'summaries': summaries, 'new_responses': calls, 'new_decisions': native['decisions_replayed'], 'new_queries': native['queries_reconstructed'],
              'new_updates': 0, 'new_checkpoints': 0, 'reused_responses': 0, 'autonomous_episodes': 288, 'environment': 'native_synthetic_three_phase',
              'generation_seconds': sum(times), 'max_generation_seconds': max(times), 'errors': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__': main()
