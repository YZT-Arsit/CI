import collections
import json
import sys
from factor_data import R, read, rows, sha, verify
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env
plan = verify(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
lengths = {}; pairs = collections.defaultdict(list); clutter = collections.defaultdict(list)
for mode in ['binary', 'proof', 'echo']:
    prefix = 'echo_' if mode == 'echo' else ''; obs = rows(R / (prefix + 'observations.jsonl')); labels = rows(R / (prefix + 'labels.jsonl'))
    key = 'proof_prompt' if mode == 'proof' else 'prompt'
    lengths[mode] = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': o[key]}], tokenize=True, add_generation_prompt=True)) for o in obs]
    for i, (o, l) in enumerate(zip(obs, labels)):
        if mode != 'echo':
            value = (lengths[mode][i], max(lengths[mode][i // 4 * 4:i // 4 * 4 + 4]))
            pairs[mode, o['seed'], o['known'], o['noise'], o['vocab']].append(value)
            if o['noise'] != 'none': clutter[mode, o['seed'], o['known'], o['vocab'], o['side']].append(value)
        for word in l['valid_witnesses']: assert len(tokenizer.encode(' ' + word, add_special_tokens=False)) == 1
assert len(pairs) == 288 and all(len(v) == 2 and len(set(v)) == 1 for v in pairs.values())
assert len(clutter) == 192 and all(len(v) == 2 and len(set(v)) == 1 for v in clutter.values())
assert max(max(v) for v in lengths.values()) < 1024
result = {'lengths': lengths, 'equal_own_and_padded_pairs': len(pairs), 'equal_count_clutter_groups': len(clutter),
          'max_lengths': {m: max(v) for m, v in lengths.items()}, 'model_loaded': False, 'model_calls': 0,
          'plan_sha256': sha(R / 'PLAN.json'), 'script_sha256': sha(R / 'token_preflight.py')}
assert not (R / 'TOKEN_PREFLIGHT.json').exists()
(R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'lengths'}))
