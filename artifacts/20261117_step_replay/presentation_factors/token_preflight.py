"""Token-length checks across all four factor conditions, without a model."""
import collections
import json
import sys
from factor_data import R, rows, sha, check
sys.path.insert(0, str(R.parents[1] / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env
check(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
obs = rows(R / 'observations.jsonl'); lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': o['prompt']}], tokenize=True, add_generation_prompt=True)) for o in obs]
pairs = collections.defaultdict(list); factors = collections.defaultdict(list)
for i, o in enumerate(obs):
    factors[o['seed'], o['world_id']].append(lengths[i])
    if o['kind'] == 'switch': pairs[o['seed'], o['view'], o['pair']].append((lengths[i], max(lengths[(i // 4) * 4:(i // 4) * 4 + 4])))
assert len(pairs) == 72 and all(len(v) == 2 and len(set(v)) == 1 for v in pairs.values())
assert len(factors) == 48 and all(len(v) == 4 and len(set(v)) == 1 for v in factors.values())
assert max(lengths) < 1024
result = {'equal_pair_own_and_padded_lengths': 72, 'equal_four_condition_lengths': 48, 'max_prompt_tokens': max(lengths),
          'model_loaded': False, 'model_calls': 0, 'plan_sha256': sha(R / 'PLAN.json'), 'script_sha256': sha(R / 'token_preflight.py')}
assert not (R / 'TOKEN_PREFLIGHT.json').exists()
(R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))
