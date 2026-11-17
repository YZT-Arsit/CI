"""Descriptive matched initial-state analysis; no added model calls or gates."""
import collections
import json
from rehearsal_data import R, ROOT, read, rows, sha, verify, targets, environment


def main():
    declaration = read(R / 'INITIAL_DIAGNOSTIC_PLAN.json')
    assert declaration['script_sha256'] == sha(R / 'diagnose_initial.py') and declaration['plan_sha256'] == sha(R / 'PLAN.json')
    assert declaration['source_sha256'] == sha(ROOT / declaration['source']) and declaration['new_model_calls'] == 0
    assert all(sha(R / name) == digest for name, digest in declaration['preserved_revision'].items())
    plan = verify(); seal = read(R / 'SEALED.json'); assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(R / name) == digest for name, digest in seal['files'].items())
    audited = read(R / 'POLICY_RESULTS.json'); assert audited['errors'] == 0 and audited['autonomous_episodes'] == 288
    worlds = read(R / 'dev_worlds.json'); summaries = []; expected_keys = None
    for arm in plan['arms']:
        counts = collections.Counter(); correct = {}; keys = []; pairs = []
        for condition, ws in worlds.items():
            raw = rows(R / 'study' / arm / (condition + '.jsonl')); assert len(raw) == len(ws) == 2
            for world, row in zip(ws, raw):
                env = environment(world); obs = env.observation(env.initial()); case = next(c for c in targets(world) if c['observation'] == obs)
                entry = row['trace'][0]; assert entry['phase'] == 0 and entry['role'] == obs['role']
                assert row['world_id'] == world['world_id'] and len(entry['queries']) == 2
                labels = {c['action']: str(c['safe_completion']) for c in case['candidates']}
                qok = all(q['estimate'] == labels[q['action']] for q in entry['queries'])
                aok = entry['selected'] in [case['candidates'][i]['action'] for i in case['optimal_indices']]
                counts.update(states=1, all_queries_correct=int(qok), optimal_action=int(aok), joint_correct=int(qok and aok))
                counts['judgments_correct_action_wrong'] += qok and not aok
                counts['judgments_wrong_action_correct'] += not qok and aok
                key = condition, world['world_id']; keys.append(key); correct[condition, world['base_group_id']] = (qok, aok)
        if expected_keys is None: expected_keys = keys
        else: assert keys == expected_keys
        for condition in plan['paired_conditions']:
            count = collections.Counter()
            for world in worlds[condition]:
                a = correct[condition, world['base_group_id']]; b = correct[condition + '_swap', world['base_group_id']]
                count.update(pairs=1, both_judgments_correct=int(a[0] and b[0]), both_actions_optimal=int(a[1] and b[1]), joint_pair_correct=int(all(a + b)))
            pairs.append({'condition': condition, **dict(count)})
        assert counts['states'] == 32
        summaries.append({'arm': arm, **dict(counts), 'matched_changing_pairs': pairs})
    result = {'summaries': summaries, 'new_model_calls': 0, 'matched_initial_states_per_model': 32,
              'scope': 'Post-declared descriptive initial-state decomposition, identical states across nine models. Judgments/action association does not establish causal use of judgments.',
              'results_sha256': sha(R / 'POLICY_RESULTS.json'), 'script_sha256': sha(R / 'diagnose_initial.py'),
              'declaration_sha256': sha(R / 'INITIAL_DIAGNOSTIC_PLAN.json')}
    (R / 'INITIAL_DIAGNOSTICS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__': main()
