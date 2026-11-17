"""Count generation batches once despite repeated per-response batch timings."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
plan = json.loads((R / 'PLAN.json').read_text())
receipts = []
for name in ['LEARNED_SEALED.json', 'ORACLE_SEALED.json']:
    seal = json.loads((R / name).read_text())
    assert seal['complete']
    for filename, digest in seal['files'].items():
        path = R / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        rows = list(map(json.loads, path.read_text().splitlines()))
        query = filename.startswith('queries_')
        elapsed = []
        for start in range(0, len(rows), plan['batch_states']):
            batch = rows[start:start + plan['batch_states']]
            timings = [q['seconds_batch'] for row in batch for q in row['queries']] if query else [row['seconds_batch'] for row in batch]
            assert len(set(timings)) == 1
            elapsed.append(timings[0])
        receipts.append({'file': filename, 'batches': len(elapsed), 'seconds': sum(elapsed), 'max_seconds': max(elapsed),
                         'responses': sum(len(row['queries']) for row in rows) if query else len(rows)})
result = {'files': receipts, 'main_generation_batches': sum(row['batches'] for row in receipts),
          'main_generated_responses': sum(row['responses'] for row in receipts),
          'generation_seconds_sum': sum(row['seconds'] for row in receipts),
          'max_generation_seconds': max(row['max_seconds'] for row in receipts),
          'preflight_excluded_responses': 13, 'preflight_excluded_batches': 2,
          'excludes_model_loading_tokenization_preparation_and_auditing': True, 'optimizer_updates': 0}
assert result['main_generation_batches'] == 2260 and result['main_generated_responses'] == 11392
(R / 'COSTS.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
