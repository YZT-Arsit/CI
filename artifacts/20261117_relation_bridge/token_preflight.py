import json
import sys
from bridge_data import R, read, rows, sha, verify
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env
plan = verify(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
def encode(text): return tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True)
lengths = {m: [] for m in plan['modes']}; changed = []; masked_changes = []
for q in rows(R / 'queries.jsonl'):
    tokens = {m: encode(q['prompts'][m]) for m in plan['modes']}
    for m in plan['modes']: lengths[m].append(len(tokens[m]))
    a, b = tokens['truth'], tokens['flip']; assert len(a) == len(b)
    changed.append(sum(x != y for x, y in zip(a, b))); assert changed[-1] == 1
    assert len(a) == len(tokens['masked'])
    masked_changes.append(sum(x != y for x, y in zip(a, tokens['masked']))); assert masked_changes[-1] == 1
actor_max = 0
for s in rows(R / 'states.jsonl'):
    text = s['actor_prefix'] + 'Own estimates (1 means compliant completion remains possible): '
    text += '; '.join(chr(65 + i) + '=unknown' for i in range(len(s['actions']))) + '.\n\nReply ANSWER:\nletter.'
    actor_max = max(actor_max, len(encode(text)))
assert max(actor_max, *(max(v) for v in lengths.values())) < 1024
assert not (R / 'TOKEN_PREFLIGHT.json').exists()
result = {'query_lengths': lengths, 'equal_token_fact_pairs': len(changed), 'changed_token_positions_per_pair': sorted(set(changed)),
          'equal_token_masked_pairs': len(masked_changes), 'changed_token_positions_masked': sorted(set(masked_changes)),
          'actor_max': actor_max, 'plan_sha256': sha(R / 'PLAN.json'), 'script_sha256': sha(R / 'token_preflight.py'), 'model_calls': 0}
(R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'query_lengths'}))
