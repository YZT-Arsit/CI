"""Score fixed identical states after sealing; preserve oracle qualification."""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(R.parent / '20261116_relation_solution')]
from audit_data import check
from kci.relation_language import prompt, seed_for


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    plan = json.loads((R / 'PLAN.json').read_text())
    for name, digest in plan['files'].items():
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == digest
    seal = json.loads((R / 'RESPONSES_SEALED.json').read_text())
    assert seal['complete'] and not seal['labels_scored']
    assert all(hashlib.sha256((R / p).read_bytes()).hexdigest() == h for p, h in seal['files'].items())
    run = json.loads((R / 'RUN_PLAN.json').read_text())
    assert hashlib.sha256((R / 'probe.py').read_bytes()).hexdigest() == run['source_sha256']
    for arm, checkpoint in run['checkpoints'].items():
        folder = R.parent / '20261117_binding_factors/repair' / arm
        assert checkpoint == json.loads((folder / 'CHECKPOINT_SEALED.json').read_text())
        assert all(hashlib.sha256((folder / 'adapter' / p).read_bytes()).hexdigest() == h
                   for p, h in checkpoint['adapter_hashes'].items())
    cases = {row['case_id']: row for row in rows(R / 'observations.jsonl')}
    labels = {row['case_id']: row for row in rows(R / 'labels.jsonl')}
    assert cases.keys() == labels.keys() and len(cases) == 560
    for key, target in labels.items():
        assert cases[key]['observation'] == target['observation'] and not check(target)
        values = [(c['immediate_safe'], c['safe_completion'], 0) for c in target['candidates']]
        assert values == [tuple(c['value']) for c in target['candidates']]
        assert target['optimal_indices'] == [i for i, v in enumerate(values) if v == max(values)]
    pool = json.loads((R.parent / '20261116_relation_solution/entity_palettes.json').read_text())['fresh']
    metrics = defaultdict(Counter)
    scored = []
    maximum_seconds = 0.
    for arm in plan['arms']:
        records = rows(R / (arm + '.jsonl'))
        assert len(records) == len({r['case_id'] for r in records}) == len(cases)
        for record in records:
            case = cases[record['case_id']]
            target = labels[record['case_id']]
            obs = case['observation']
            nonce = seed_for(case['base_group_id'], {'Planner': 0, 'Source': 1, 'Worker': 2}[obs['role']])
            text, order = prompt(obs, pool, nonce)
            assert record['actions'] == order
            estimates = {}
            all_correct = True
            family = re.sub(r'_v[01]', '', case['condition']).removesuffix('_swap')
            stats = metrics[(arm, family, obs['role'])]
            assert [q['action'] for q in record['queries']] == obs['actions']
            for q in record['queries']:
                qtext = prompt(obs, pool, nonce, question='task', candidate=q['action'])[0]
                qtext = qtext.replace('A yes; B no.', 'Reply ANSWER: followed by 1 if possible, otherwise 0.')
                assert q['prompt'] == qtext
                found = re.findall(r'ANSWER:\s*([01])\b', q['raw'])
                valid = len(found) == 1 and q['seconds_batch'] < 9.5
                bit = found[0] if valid else 'unknown'
                assert valid == q['valid'] and bit == q['estimate']
                truth = str(next(c['safe_completion'] for c in target['candidates'] if c['action'] == q['action']))
                estimates[q['action']] = bit
                stats['queries'] += 1
                stats['query_correct'] += bit == truth
                stats['query_truth_' + truth] += 1
                stats['query_correct_truth_' + truth] += bit == truth
                stats['invalid_queries'] += not valid
                all_correct &= bit == truth
                maximum_seconds = max(maximum_seconds, q['seconds_batch'])
            text += 'Own estimates (1 means compliant completion remains possible): ' + '; '.join(chr(65 + i) + '=' + estimates[a] for i, a in enumerate(order)) + '.\n'
            assert record['prompt'] == text + '\nReply ANSWER:\nletter.'
            found = re.findall(r'ANSWER:\s*([ABC])\b', record['raw'])
            valid = len(found) == 1 and ord(found[0]) - 65 < len(order) and record['seconds_batch'] < 9.5
            selected = order[ord(found[0]) - 65] if valid else None
            assert record['valid'] == valid and record['selected'] == selected
            stats['selected_letter_' + (found[0] if valid else 'INVALID')] += 1
            best = {target['candidates'][i]['action'] for i in target['optimal_indices']}
            optimal = selected in best
            safe = valid and bool(next(c['immediate_safe'] for c in target['candidates'] if c['action'] == selected))
            stats.update({'states': 1, 'all_estimates_correct': int(all_correct), 'optimal_actions': int(optimal),
                          'immediate_safe_actions': int(safe), 'invalid_actions': int(not valid),
                          'wrong_action_despite_all_correct_estimates': int(all_correct and not optimal),
                          'wrong_action_with_incorrect_estimate_map': int(not all_correct and not optimal),
                          'optimal_action_with_incorrect_estimate_map': int(not all_correct and optimal),
                          'withhold': int(selected == 'WITHHOLD'),
                          'withhold_when_safe_completion_possible': int(selected == 'WITHHOLD' and not target['no_safe_completion'])})
            maximum_seconds = max(maximum_seconds, record['seconds_batch'])
            scored.append({'arm': arm, 'case_id': case['case_id'], 'condition': case['condition'],
                           'state_id': case['state_id'], 'role': obs['role'], 'estimates': estimates,
                           'all_estimates_correct': all_correct, 'optimal': optimal, 'safe': safe,
                           'selected': selected, 'valid': valid})
    indexed = {(r['arm'], r['case_id']): r for r in scored}
    matched = defaultdict(Counter)
    for key, case in cases.items():
        a = indexed[('two_view_sft', key)]
        b = indexed[('conditional_js', key)]
        family = re.sub(r'_v[01]', '', case['condition']).removesuffix('_swap')
        stratum = ('correct' if a['all_estimates_correct'] else 'incorrect') + '_vs_' + ('correct' if b['all_estimates_correct'] else 'incorrect')
        stats = matched[(family, a['role'], stratum)]
        stats.update({'states': 1, 'sft_optimal': int(a['optimal']), 'js_optimal': int(b['optimal']),
                      'sft_safe': int(a['safe']), 'js_safe': int(b['safe']),
                      'same_estimate_maps': int(a['estimates'] == b['estimates'])})
    constant = defaultdict(Counter)
    for target in labels.values():
        obs = target['observation']
        tickets = {p['ticket']: str(i) for i, p in enumerate(obs['contract']['providers'])}
        counts = constant[obs['role']]
        counts['states'] += 1
        for i in target['optimal_indices']:
            action = target['candidates'][i]['action']
            if action.startswith('COMMIT:'):
                action = 'COMMIT_provider_index_' + tickets[action.split(':', 1)[1]]
            counts[action] += 1
    result = {'groups': [{'arm': a, 'family': f, 'role': role, **dict(v)} for (a, f, role), v in metrics.items()],
              'matched_strata': [{'family': f, 'role': role, 'estimate_correctness_sft_vs_js': s, **dict(v)}
                                for (f, role, s), v in matched.items()],
              'constant_action_diagnostics': {role: dict(v) for role, v in constant.items()},
              'constant_action_note': 'Exhaustive finite label counts, not fitted baseline or relational-learning evidence. Planner actions normalized by provider index.',
              'state_versions_per_arm': 560, 'unique_states_per_arm': 280, 'max_generation_seconds': maximum_seconds,
              'errors': 0, 'scope': 'Matched fixed-state DEV diagnosis, not new test or autonomous trajectories',
              'whole_goal_complete': False}
    (R / 'SCORED.json').write_text(json.dumps(scored), encoding='utf-8')
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['groups', 'matched_strata']}))


if __name__ == '__main__':
    main()
