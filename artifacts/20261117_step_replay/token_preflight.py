"""Verify identical local tokens, actor token positions and paired eval widths."""
import collections
import json
import sys
from replay_data import R, PARENT, NAMES, read, rows, sha, verify, prompt
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env


plan = verify(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
pool = read(NAMES / 'palettes.json')['train']; data = read(R / 'training_worlds.json'); totals = {}; maximum = 0
def length(text):
    return len(tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True))
for character in '01ABC':
    encoded = tokenizer.encode('ANSWER:\n' + character + tokenizer.eos_token, add_special_tokens=False)
    assert encoded[-2] == tokenizer.encode(character, add_special_tokens=False)[0] and encoded[-1] == tokenizer.eos_token_id
for seed in plan['training_seeds']:
    total = []
    for local, replay in zip(rows(R / f'local_{seed}.jsonl'), rows(R / f'replay_{seed}.jsonl')):
        texts = [prompt(data[key], pool, n) for key, n in zip(local['ids'], local['names'])]
        answers = ['ANSWER:\n' + str(bit) for bit in local['labels']]
        groups = []
        for ts, ans in [(texts, answers), ([e['prompt'] for e in replay['examples']], [e['answer'] for e in replay['examples']])]:
            lengths = [length(t) for t in ts]; maximum = max(maximum, max(lengths))
            groups.append(sum(n + len(tokenizer.encode(a + tokenizer.eos_token, add_special_tokens=False)) for n, a in zip(lengths, ans)))
        total.append({'local_tokens': groups[0], 'actor_tokens': groups[1]})
    assert [t['local_tokens'] for t in total] == read(PARENT / 'TOKEN_PREFLIGHT.json')['training_tokens_by_step'][str(seed)]
    totals[str(seed)] = total
obs = rows(R / 'observations.jsonl'); lengths = [length(o['prompt']) for o in obs]; pairs = collections.defaultdict(list)
for i, o in enumerate(obs):
    if o['kind'] == 'switch': pairs[o['group'], o['vocab'], o['view']].append((lengths[i], max(lengths[(i // 4) * 4:(i // 4) * 4 + 4])))
assert len(pairs) == 128 and all(len(v) == 2 and len(set(v)) == 1 for v in pairs.values())
assert max(maximum, max(lengths)) < 1024
result = {'training_tokens_by_step': totals, 'training_max_prompt_tokens': maximum, 'confirmation_max_prompt_tokens': max(lengths),
          'equal_own_and_padded_pairs': 128, 'local_training_tokens_match_prior': True, 'answer_positions_verified': True,
          'model_loaded': False, 'model_calls': 0, 'script_sha256': sha(R / 'token_preflight.py'), 'plan_sha256': sha(R / 'PLAN.json')}
assert not (R / 'TOKEN_PREFLIGHT.json').exists()
(R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'training_tokens_by_step'}))
