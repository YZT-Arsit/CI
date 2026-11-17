import json
import sys
from join_handoff_data import H, R, read, rows, sha, verify
import run_handoff
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env
plan = verify(); assert not (H / 'TOKEN_PREFLIGHT.json').exists()
tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True); obs = rows(H / 'observations.jsonl')
tokens = [tokenizer.apply_chat_template([{'role': 'user', 'content': o['prompt']}], tokenize=True, add_generation_prompt=True) for o in obs]
for i in range(0, 32, 2):
    assert obs[i]['state_id'] == obs[i + 1]['state_id'] and len(tokens[i]) == len(tokens[i + 1])
    assert sum(a != b for a, b in zip(tokens[i], tokens[i + 1])) == 2
assert max(map(len, tokens)) < 1024 and run_handoff.runner.R == R
result = {'plan_sha256': sha(H / 'PLAN.json'), 'script_sha256': sha(H / 'token_preflight.py'), 'lengths': list(map(len, tokens)),
          'equal_token_pairs': 16, 'changed_token_positions_per_pair': 2, 'model_loaded': False, 'model_calls': 0}
(H / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))
