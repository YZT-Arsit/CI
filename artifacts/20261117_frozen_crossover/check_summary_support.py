"""Exercise a three-way incompatibility with no disjoint pair of optima."""
import json
from pathlib import Path
from audit_summary_support import group

records = []
for index, best in enumerate([{0, 1}, {1, 2}, {0, 2}]):
    records.append({'case_id': str(index), 'state_id': str(index),
                    'observation': {'role': 'synthetic_control'}, 'optimal_indices': sorted(best),
                    'candidates': [{'action': action, 'safe_completion': 0, 'immediate_safe': int(i in best)}
                                   for i, action in enumerate(['A', 'B', 'C'])]})
result = group(records, ['safe_completion'])
assert result['incompatible_groups'] == 1 and len(result['collisions'][0]['witnesses']) == 3
assert group(records, ['immediate_safe', 'safe_completion'])['incompatible_groups'] == 0
assert group(records[:2], ['safe_completion'])['incompatible_groups'] == 0
receipt = {'three_way_conflict_detected': True, 'pairwise_compatible_control_passed': True,
           'full_value_summary_control_passed': True, 'synthetic_grouping_test_only': True, 'model_calls': 0}
(Path(__file__).resolve().parent / 'SUMMARY_SUPPORT_CHECK.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
