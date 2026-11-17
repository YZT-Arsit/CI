"""Register exact TRAIN query-fit diagnosis before main result scoring."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert not (R / 'TRAIN_QUERY_PLAN.json').exists() and not (R / 'RESULTS.json').exists()
plan = json.loads((R / 'PLAN.json').read_text())
assert sha(R / 'train_oracle.jsonl') == plan['data_hashes']['train_oracle.jsonl']
rows = list(map(json.loads, (R / 'train_oracle.jsonl').read_text().splitlines()))
for row in rows:
    assert [c['action'] for c in row['candidates']] == row['observation']['actions']
observations = [{key: row[key] for key in ['case_id', 'step', 'seeds', 'observation']} for row in rows]
path = R / 'train_observations.jsonl'
path.write_text(''.join(json.dumps(row) + '\n' for row in observations))
spec = {'arms': ['own_sft', 'teacher_sft'], 'parent_plan_sha256': sha(R / 'PLAN.json'),
        'observations_sha256': sha(path), 'labels_sha256': sha(R / 'train_oracle.jsonl'),
        'source_hashes': {name: sha(R / name) for name in ['train_queries.py', 'audit_train_queries.py', 'build_train_queries.py']},
        'presentations_per_arm': 320, 'query_responses_per_arm': 800, 'model_updates': 0,
        'registered_before_main_scoring': True, 'read_labels_in_worker': False,
        'scope': 'Exact TRAIN query-prompt diagnostic on the same two preselected models. Distinguish training fit from DEV generalization; no new validation cases or optimization.'}
(R / 'TRAIN_QUERY_PLAN.json').write_text(json.dumps(spec, indent=2))
print(json.dumps(spec))
