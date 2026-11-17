"""Reconstruct sealed fixed-state outputs, native rollouts and matched interventions."""
import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(R.parent / '20261116_relation_solution'),
                str(R.parent / '20261117_binding_factors'), str(R.parent / '20261117_sampler_repair')]
from audit_data import check
from experiments.relation_solution_data import environment, targets
from kci.relation_language import prompt, seed_for
from render import actor as training_actor
from sampling import buckets


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return list(map(json.loads, path.read_text(encoding='utf-8').splitlines()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonce(case):
    return seed_for(case['base_group_id'], {'Planner': 0, 'Source': 1, 'Worker': 2}[case['observation']['role']])


def key(condition, world, observation):
    return condition, world, json.dumps(observation, sort_keys=True)


def main():
    plan = read(R / 'PLAN.json')
    scoring = read(R / 'SCORING_PLAN.json')
    assert scoring['auditor_sha256'] == sha(Path(__file__))
    assert scoring['prior_train_plan_sha256'] == sha(R.parent / '20261117_teacher_binding/PLAN.json')
    assert scoring['prior_results_sha256'] == sha(R.parent / '20261117_teacher_binding/repair/RESULTS.json')
    assert scoring['preflight_sha256'] == sha(R / 'PREFLIGHT.json')
    assert read(R / 'RUN_PLAN.json')['plan_sha256'] == sha(R / 'PLAN.json')
    for name, digest in {**plan['source_hashes'], **plan['data_hashes']}.items():
        assert sha(R / name) == digest
    assert all(sha(ROOT / name) == digest for name, digest in plan['dependency_hashes'].items())
    for seal_name in ['QUERIES_SEALED.json', 'LEARNED_SEALED.json', 'ORACLE_SEALED.json']:
        seal = read(R / seal_name)
        assert seal['complete'] and not seal['labels_scored']
        assert all(sha(R / name) == digest for name, digest in seal['files'].items())
    assert not read(R / 'RUN_PLAN.json')['read_labels']
    oracle_plan = read(R / 'ORACLE_RUN_PLAN.json')
    assert oracle_plan['read_labels'] and not oracle_plan['learned_policy_evidence']
    assert oracle_plan['plan_sha256'] == sha(R / 'PLAN.json') and oracle_plan['learned_seal_sha256'] == sha(R / 'LEARNED_SEALED.json')
    for item in plan['checkpoints'].values():
        folder = ROOT / item['path']
        assert read(folder / 'CHECKPOINT_SEALED.json') == item['seal']
        assert all(sha(folder / 'adapter' / name) == digest for name, digest in item['seal']['adapter_hashes'].items())
    worlds_path = ROOT / plan['source_dev_path']
    assert sha(worlds_path) == plan['source_dev_sha256']
    worlds = read(worlds_path)
    cases_list = rows(R / 'observations.jsonl')
    cases = {case['case_id']: case for case in cases_list}
    labels = {case['case_id']: case for case in rows(R / 'labels.jsonl')}
    assert cases.keys() == labels.keys() and len(cases) == len(cases_list) == 560
    lookup = {}
    for condition, group in worlds.items():
        for world in group:
            for target in targets(world):
                cid = hashlib.sha256((condition + ':' + target['state_id']).encode()).hexdigest()
                assert labels[cid] == {'case_id': cid, **target} and not check(target)
                assert cases[cid] == {'case_id': cid, 'condition': condition, **{k: target[k] for k in
                    ['state_id', 'world_id', 'base_group_id', 'observation']}}
                lookup[key(condition, world['world_id'], target['observation'])] = cid
    assert len(lookup) == 560
    pools = read(R.parent / '20261116_relation_solution/entity_palettes.json')
    estimates, query_valid, query_records = {}, {}, {}
    query_stats = defaultdict(Counter)
    max_seconds = 0.
    for source in plan['arms']:
        records = rows(R / f'queries_{source}.jsonl')
        assert [r['case_id'] for r in records] == list(cases)
        for record in records:
            cid = record['case_id']; case = cases[cid]; obs = case['observation']
            assert [q['action'] for q in record['queries']] == obs['actions']
            bits = {}; valid_map = True
            truth = {c['action']: str(c['safe_completion']) for c in labels[cid]['candidates']}
            for q in record['queries']:
                text = prompt(obs, pools['fresh'], nonce(case), question='task', candidate=q['action'])[0]
                assert q['prompt'] == text.replace('A yes; B no.', 'Reply ANSWER: followed by 1 if possible, otherwise 0.')
                found = re.findall(r'ANSWER:\s*([01])\b', q['raw'])
                valid = len(found) == 1 and q['seconds_batch'] < 9.5
                bit = found[0] if valid else 'unknown'
                assert q['valid'] == valid and q['estimate'] == bit
                bits[q['action']] = bit; valid_map &= valid
                query_stats[(source, obs['role'])].update(queries=1, correct=int(bit == truth[q['action']]), invalid=int(not valid))
                max_seconds = max(max_seconds, q['seconds_batch'])
            estimates[source, cid] = bits; query_valid[source, cid] = valid_map
            query_records[source, cid] = record['queries']
            query_stats[(source, obs['role'])].update(states=1, all_correct=int(bits == truth))
    choices = {}; metrics = defaultdict(Counter); scored = []
    for actor in plan['arms']:
        for source in plan['arms'] + ['oracle']:
            path = R / (f'oracle_dev_{actor}.jsonl' if source == 'oracle' else f'actor_{actor}__query_{source}.jsonl')
            records = rows(path)
            assert [r['case_id'] for r in records] == list(cases)
            for record in records:
                cid = record['case_id']; case = cases[cid]; target = labels[cid]; obs = case['observation']
                truth = {c['action']: str(c['safe_completion']) for c in target['candidates']}
                bits = truth if source == 'oracle' else estimates[source, cid]
                text, order = prompt(obs, pools['fresh'], nonce(case))
                text += 'Own estimates (1 means compliant completion remains possible): ' + '; '.join(chr(65+i)+'='+bits[action] for i, action in enumerate(order)) + '.\n\nReply ANSWER:\nletter.'
                assert record['prompt'] == text and record['actions'] == order and record['input_estimates'] == bits
                found = re.findall(r'ANSWER:\s*([ABC])\b', record['raw'])
                valid = len(found) == 1 and ord(found[0])-65 < len(order) and record['seconds_batch'] < 9.5
                selected = order[ord(found[0])-65] if valid else None
                assert record['valid'] == valid and record['selected'] == selected
                best = {target['candidates'][i]['action'] for i in target['optimal_indices']}
                optimal = selected in best
                safe = valid and next(c['immediate_safe'] for c in target['candidates'] if c['action'] == selected)
                family = re.sub(r'_v[01]', '', case['condition']).removesuffix('_swap')
                metrics[(actor, source, family, obs['role'])].update(states=1, optimal=int(optimal), safe=int(safe),
                    invalid=int(not valid), all_estimates_correct=int(bits == truth),
                    wrong_despite_correct_estimates=int(bits == truth and not optimal),
                    wrong_with_multiple_optima=int(not optimal and len(best) > 1))
                choices[actor, source, cid] = record
                scored.append({'actor': actor, 'source': source, 'case_id': cid, 'optimal': optimal, 'safe': bool(safe),
                               'role': obs['role'], 'family': family, 'number_of_optimal_actions': len(best)})
                max_seconds = max(max_seconds, record['seconds_batch'])
    indexed_scores = {(s['actor'], s['source'], s['case_id']): s for s in scored}
    interventions = []
    for actor in plan['arms']:
        for left, right in itertools.combinations(plan['arms'] + ['oracle'], 2):
            count = Counter()
            for cid in cases:
                a, b = choices[actor, left, cid], choices[actor, right, cid]
                sa, sb = indexed_scores[actor, left, cid], indexed_scores[actor, right, cid]
                same = a['prompt'] == b['prompt']
                changed = (a['valid'], a['selected']) != (b['valid'], b['selected'])
                count.update(states=1, identical_prompts=int(same), identical_prompt_disagreements=int(same and changed),
                             changed_prompts=int(not same), changed_actions=int(not same and changed),
                             repaired_errors=int(not sa['optimal'] and sb['optimal']),
                             introduced_errors=int(sa['optimal'] and not sb['optimal']))
            interventions.append({'actor': actor, 'left_source': left, 'right_source': right, **dict(count)})
    rollout_records = []; rollout_summary = []
    for actor in plan['arms']:
        for source in plan['arms'] + ['oracle']:
            by_condition = {}
            for condition, group in worlds.items():
                episodes = []
                for world in group:
                    env = environment(world); state = env.initial(); invalid = False; bad_query = False; trace = []
                    for phase in range(3):
                        if state.done or invalid:
                            break
                        cid = lookup[key(condition, world['world_id'], env.observation(state))]
                        record = choices[actor, source, cid]
                        if source != 'oracle':
                            bad_query |= not query_valid[source, cid]
                        if not record['valid']:
                            invalid = True
                        else:
                            state, verdict = env.transition(state, record['selected'])
                            invalid = verdict['verdict'] == 'UNKNOWN'
                        trace.append({'case_id': cid, 'selected': record['selected'], 'valid': record['valid']})
                    success = bool(state.success and not state.violation and not invalid and not bad_query)
                    row = {'actor': actor, 'source': source, 'condition': condition, 'world_id': world['world_id'],
                           'strict_success': success, 'valid_ci': not state.violation and not invalid,
                           'invalid': invalid, 'trace': trace, 'final': asdict(state)}
                    episodes.append(row); rollout_records.append(row)
                by_condition[condition] = episodes
            pairs = {}
            for family in ['base', 'depth3', 'internal3', 'cycle4']:
                groups = [by_condition[family + '_v' + str(v) + suffix] for v in (0, 1) for suffix in ['', '_swap']]
                pairs[family] = {'groups': len(groups[0]), 'all_four_success': sum(all(row['strict_success'] for row in group) for group in zip(*groups))}
            all_rows = [row for group in by_condition.values() for row in group]
            rollout_summary.append({'actor': actor, 'source': source, 'episodes': len(all_rows),
                'strict_success': sum(row['strict_success'] for row in all_rows),
                'valid_ci': sum(row['valid_ci'] for row in all_rows), 'pairs': pairs,
                'oracle_input': source == 'oracle', 'shared_checkpoint': source == actor})
    train = rows(R / 'train_oracle.jsonl'); data = buckets()
    original_plan = read(R.parent / '20261117_teacher_binding/PLAN.json')
    expected = []
    for step, batch in enumerate(original_plan['batches']):
        for view, seeds in enumerate(batch['views']):
            for side, case in enumerate(data[tuple(batch['key'])][batch['index']]):
                expected.append({'case_id': f'train_{step}_{view}_{side}', 'step': step, 'seeds': seeds, **case})
    assert train == expected
    train_stats = defaultdict(Counter)
    for actor in plan['train_arms']:
        records = rows(R / f'oracle_train_{actor}.jsonl')
        assert len(records) == len(train) == 320
        for case, record in zip(train, records):
            assert not check(case)
            bits = {c['action']: str(c['safe_completion']) for c in case['candidates']}
            text, order = training_actor(case['observation'], pools['train'], case['seeds'], bits)
            assert record['case_id'] == case['case_id'] and record['prompt'] == text and record['actions'] == order and record['input_estimates'] == bits
            found = re.findall(r'ANSWER:\s*([ABC])\b', record['raw'])
            valid = len(found) == 1 and ord(found[0])-65 < len(order) and record['seconds_batch'] < 9.5
            selected = order[ord(found[0])-65] if valid else None
            assert record['valid'] == valid and record['selected'] == selected
            best = {case['candidates'][i]['action'] for i in case['optimal_indices']}
            train_stats[actor, case['observation']['role']].update(presentations=1, optimal=int(selected in best), invalid=int(not valid))
            max_seconds = max(max_seconds, record['seconds_batch'])
    diagonal = defaultdict(Counter)
    prior_reports = {(row['arm'], row['condition']): row for row in
                     read(R.parent / '20261117_teacher_binding/repair/RESULTS.json')['summaries']}
    for arm in plan['arms']:
        prior_folder = ROOT / plan['checkpoints'][arm]['path']
        for condition, group in worlds.items():
            prior_path = prior_folder / (condition + '.jsonl')
            prior_rows = rows(prior_path)
            report = prior_reports[arm, condition]
            assert sha(prior_path) == report['raw_sha256'] and len(prior_rows) == len(group) == report['episodes']
            for world, prior in zip(group, prior_rows):
                assert prior['world_id'] == world['world_id'] and prior['base_group_id'] == world['base_group_id']
                env = environment(world); state = env.initial()
                for entry in prior['trace']:
                    cid = lookup[key(condition, world['world_id'], env.observation(state))]
                    new = choices[arm, arm, cid]
                    qs = query_records[arm, cid]
                    assert [q['prompt'] for q in qs] == [q['prompt'] for q in entry['queries']]
                    same_prompt = new['prompt'] == entry['prompt']
                    diagonal[arm].update(states=1, changed_query_maps=int([(q['valid'], q['estimate']) for q in qs] != [(q['valid'], q['estimate']) for q in entry['queries']]),
                        identical_actor_prompts=int(same_prompt), identical_actor_prompt_disagreements=int(same_prompt and (new['valid'], new['selected']) != (entry['format_valid'], entry['selected'])))
                    if entry['format_valid']:
                        state, _ = env.transition(state, entry['selected'])
        assert diagonal[arm]['states'] == scoring['prior_expected_decisions'][arm]
    preflight = read(R / 'PREFLIGHT.json')
    assert len(preflight['queries']) == preflight['query_responses'] == 9
    assert len(preflight['actors']) == preflight['actor_responses'] == 4
    preflight_mismatch = 0
    for query in preflight['queries']:
        actual = next(q for q in query_records['teacher_sft', query['case_id']] if q['action'] == query['action'])
        assert actual['prompt'] == query['prompt']
        preflight_mismatch += actual['raw'] != query['raw']
    for record in preflight['actors']:
        actual = choices['teacher_sft', 'teacher_sft', record['case_id']]
        assert actual['prompt'] == record['prompt']
        preflight_mismatch += actual['raw'] != record['raw']
    result = {'query_metrics': [{'source': s, 'role': role, **dict(v)} for (s, role), v in query_stats.items()],
        'action_metrics': [{'actor': a, 'source': s, 'family': f, 'role': role, **dict(v)} for (a, s, f, role), v in metrics.items()],
        'interventions': interventions, 'reconstructed_rollouts': rollout_summary,
        'train_oracle_metrics': [{'actor': a, 'role': role, **dict(v)} for (a, role), v in train_stats.items()],
        'prior_online_diagonal_agreement': {arm: dict(v) for arm, v in diagonal.items()},
        'query_responses': sum(v['queries'] for v in query_stats.values()), 'dev_actor_responses': len(scored),
        'train_oracle_responses': sum(v['presentations'] for v in train_stats.values()),
        'reconstructed_episodes': len(rollout_records), 'max_generation_seconds': max_seconds,
        'preflight_repeated_raw_responses': 13, 'preflight_raw_disagreements': preflight_mismatch,
        'auditor_sha256': sha(Path(__file__)),
        'errors': 0, 'same_DEV_reused': True, 'whole_goal_complete': False}
    assert result['query_responses'] == 4032 and result['dev_actor_responses'] == 6720 and result['train_oracle_responses'] == 640
    assert result['reconstructed_episodes'] == 1344
    (R / 'SCORED.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in scored))
    (R / 'RECONSTRUCTED_ROLLOUTS.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rollout_records))
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['query_metrics', 'action_metrics', 'interventions']}))


if __name__ == '__main__':
    main()
