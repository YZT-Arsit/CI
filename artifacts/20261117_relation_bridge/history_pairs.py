"""Descriptive actual-history pairs; no new calls or primary-gate change."""
import json
from bridge_data import R, read, rows, sha, verify


def main():
    declaration = read(R / 'HISTORY_PLAN.json'); plan = verify(); seal = read(R / 'SEALED.json')
    assert declaration['plan_sha256'] == sha(R / 'PLAN.json') and declaration['source_sha256'] == sha(R / 'history_pairs.py')
    assert seal['complete'] and all(sha(R / p) == h for p, h in seal['files'].items())
    assert read(R / 'RESULTS.json')['errors'] == 0
    queries = {q['id']: q for q in rows(R / 'queries.jsonl')}; labels = {l['id']: l for l in rows(R / 'labels.jsonl')}
    states = {s['id']: s for s in rows(R / 'states.jsonl')}; pairs = []
    for key, q in queries.items():
        state = states[q['state_id']]; family = state['condition']
        if family not in ['base', 'depth3']: continue
        other = key.replace(family + '_', family + '_swap_', 1)
        assert q['action'] == queries[other]['action']
        a, b = labels[key], labels[other]; ca, cb = a['case'], b['case']
        assert ca['observation']['role'] == cb['observation']['role']
        assert ca['observation']['contract'] == cb['observation']['contract']
        assert ca['observation']['committed_ticket'] == cb['observation']['committed_ticket']
        pairs.append((key, other, family, a['truth_target'] != b['truth_target']))
    assert len(pairs) == 48 and sum(p[3] for p in pairs) == 24
    reports = []
    for seed in plan['seeds']:
        for mode in plan['modes']:
            raw = {r['id']: r for r in rows(R / 'study' / str(seed) / (mode + '_queries.jsonl'))}
            actions = {r['id']: r for r in rows(R / 'study' / str(seed) / (mode + '_actions.jsonl'))}
            summary = {'seed': seed, 'mode': mode, 'families': []}
            for family in ['base', 'depth3']:
                selected = [p for p in pairs if p[2] == family and p[3]]
                both = sum(raw[a]['estimate'] == str(labels[a]['truth_target']) and raw[b]['estimate'] == str(labels[b]['truth_target']) for a, b, _, _ in selected)
                planner_pairs = []
                for a, b, _, _ in selected:
                    ca = labels[a]['case']; cb = labels[b]['case']
                    if ca['observation']['role'] != 'Planner' or labels[a]['candidate_index'] != 0: continue
                    good = []
                    for key, case in [(a, ca), (b, cb)]:
                        good.append(actions[labels[key]['state_id']]['selected'] in [case['candidates'][i]['action'] for i in case['optimal_indices']])
                    planner_pairs.append(all(good))
                assert len(selected) == 12 and len(planner_pairs) == 2
                summary['families'].append({'family': family, 'changing_query_pairs': 12, 'both_queries_correct_true_world': both,
                                           'planner_pairs': 2, 'both_actions_optimal_true_world': sum(planner_pairs)})
            reports.append(summary)
    result = {'summaries': reports, 'new_model_calls': 0, 'autonomous_episodes': 0, 'declaration_sha256': sha(R / 'HISTORY_PLAN.json'),
              'scope': 'Actual R-history swaps within the pre-existing frozen worlds, one alias view. All models/modes reported. Truth/flip note inputs remain external; no proof of model-generated relations.'}
    (R / 'HISTORY_RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__': main()
