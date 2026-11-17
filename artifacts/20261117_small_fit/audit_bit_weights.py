"""Post-launch analytic diagnostic of the unchanged binary CE weighting, not model behavior."""
import collections
import json
from fit_data import R, ROOT, read, sha, verify, batches


def main():
    verify()
    source = ROOT / 'artifacts/20261117_teacher_binding/binding_loss.py'
    records = []
    for index, batch in enumerate(batches()):
        pair = batch['pairs'][str(batch['mixed_depth'])]
        labels = [c['safe_completion'] for _ in batch['views'] for row in pair for c in row['candidates']]
        counts = collections.Counter(labels)
        assert set(counts) == {0, 1}
        weights = {y: .5 / counts[y] for y in [0, 1]}
        for a, b in zip(pair[0]['candidates'], pair[1]['candidates']):
            if a['safe_completion'] == b['safe_completion']:
                continue
            positive = weights[1] * len(batch['views'])
            negative = weights[0] * len(batch['views'])
            # Conditional 0/1 log odds at fixed binary probability mass.
            # This is not an arbitrary full-vocabulary raw-logit derivative.
            optimum = positive / (positive + negative)
            gradient_at_zero = .5 * (positive + negative) - positive
            records.append({'batch_index': index, 'role': batch['key'][0], 'depth': batch['mixed_depth'],
                            'kind': batch['factors'][0], 'action': a['action'], 'positive_examples': counts[1],
                            'negative_examples': counts[0], 'changing_action_positive_weight': positive,
                            'changing_action_negative_weight': negative, 'positive_to_negative_ratio': positive / negative,
                            'constant_action_optimum_conditional_positive_probability': optimum,
                            'conditional_binary_log_odds_gradient_at_zero': gradient_at_zero,
                            'uniform_ce_constant_action_optimum': .5})
    by_role = {role: {'changing_actions': sum(r['role'] == role for r in records),
                     'weight_ratios': sorted({r['positive_to_negative_ratio'] for r in records if r['role'] == role}),
                     'constant_conditional_optima': sorted({r['constant_action_optimum_conditional_positive_probability'] for r in records if r['role'] == role})}
               for role in ['Planner', 'Source', 'Worker']}
    assert by_role['Planner']['weight_ratios'] == [1.]
    assert by_role['Source']['weight_ratios'] == [3.]
    assert by_role['Worker']['weight_ratios'] == [5.]
    result = {'by_role': by_role, 'records': records, 'source_sha256': sha(source), 'plan_sha256': sha(R / 'PLAN.json'),
              'errors': 0, 'model_calls': 0, 'changes_to_running_experiment': False,
              'scope': 'Analytic bit-loss property for hypothetical relation-blind conditional 0/1 log odds at fixed binary probability mass. Actual CE uses full-vocabulary softmax; equal raw bit logits do not imply unconditional p1=.5 when OTHER mass remains. Excludes format loss, actor loss, parameter sharing and optimizer dynamics. Does not establish that a trained model attains this optimum or that this explains every error.'}
    (R / 'BIT_WEIGHT_DIAGNOSTIC.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
