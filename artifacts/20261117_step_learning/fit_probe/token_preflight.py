"""Tokenizer-only pairing and batch-width check for the frozen fit probe."""
import collections
import json
import sys
from probe import R, rows, sha, check
sys.path.insert(0, str(R.parents[1] / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env


check()
tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
obs = rows(R / 'cases.jsonl'); pairs = collections.defaultdict(list)
lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': o['prompt']}], tokenize=True, add_generation_prompt=True)) for o in obs]
for i, o in enumerate(obs):
    if o['kind'] == 'switch':
        pairs[o['seed'], o['view'], o['pair']].append((lengths[i], max(lengths[(i // 4) * 4:(i // 4) * 4 + 4])))
assert len(pairs) == 54 and all(len(v) == 2 and len(set(v)) == 1 for v in pairs.values())
assert max(lengths) < 1024
result = {'paired_groups_equal_own_and_padded_lengths': 54, 'max_prompt_tokens': max(lengths),
          'model_loaded': False, 'model_calls': 0, 'plan_sha256': sha(R / 'PLAN.json'), 'script_sha256': sha(R / 'token_preflight.py')}
assert not (R / 'TOKEN_PREFLIGHT.json').exists()
(R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
