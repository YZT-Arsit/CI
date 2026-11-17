"""Preserve the first replay-audit failure and inspect native/JSON type equality."""
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parents[1] / 'code'))
from experiments.relation_solution_data import targets

plan = json.loads((R / 'PLAN.json').read_text())
worlds = json.loads((R.parents[1] / plan['source_dev_path']).read_text())
labels = {row['case_id']: row for row in map(json.loads, (R / 'labels.jsonl').read_text().splitlines())}
condition, values = next(iter(worlds.items()))
target = targets(values[0])[0]
cid = hashlib.sha256((condition + ':' + target['state_id']).encode()).hexdigest()
native = {'case_id': cid, **target}
actual = labels[cid]
result = {'failure': 'audit.py first native-label equality assertion failed before output scoring',
          'auditor_sha256_at_failure': hashlib.sha256((R / 'audit.py').read_bytes()).hexdigest(),
          'first_case_id': cid, 'differing_top_level_fields': [key for key in native if native[key] != actual[key]],
          'native_candidate_value_type': type(target['candidates'][0]['value']).__name__,
          'json_candidate_value_type': type(actual['candidates'][0]['value']).__name__,
          'equal_after_json_roundtrip': json.loads(json.dumps(native)) == actual,
          'model_outputs_changed': False}
(R / 'FIRST_AUDIT_FAILURE.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
