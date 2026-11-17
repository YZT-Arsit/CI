"""Cross-check actual update tokens against the tokenizer-only receipt."""
import json
from learning_data import R, read, rows, sha, verify


verify(); seal = read(R / 'SEALED.json')
assert seal['complete'] and all(sha(R / n) == h for n, h in seal['files'].items())
receipt = read(R / 'TOKEN_PREFLIGHT.json')
assert receipt['plan_sha256'] == sha(R / 'PLAN.json') and receipt['script_sha256'] == sha(R / 'token_preflight.py')
stats = []
for seed in [17, 29, 41]:
    records = rows(R / f'learning/seed{seed}/training.jsonl')
    assert [r['tokens'] for r in records] == receipt['training_tokens_by_step'][str(seed)]
    stats.append({'schedule': seed, 'updates': len(records), 'tokens': sum(r['tokens'] for r in records),
                  'first_24_bit_loss_mean': sum(r['bit_loss'] for r in records[:24]) / 24,
                  'last_24_bit_loss_mean': sum(r['bit_loss'] for r in records[-24:]) / 24})
result = {'schedules': stats, 'all_432_token_totals_verified': True, 'model_calls': 0,
          'script_sha256': sha(R / 'training_check.py'), 'seal_sha256': sha(R / 'SEALED.json')}
(R / 'TRAINING_CHECK.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
