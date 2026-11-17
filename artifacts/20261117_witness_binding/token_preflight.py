"""Full scheduled target-position and paired-length checks, tokenizer only."""
import collections
import json
import sys
from graph_data import R, NAMES, read, rows, sha, verify, render
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env
plan = verify(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
pool = read(NAMES / 'palettes.json')['train']; totals = {}; maximum = 0
def length(text): return len(tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True))
def suffix(answer): return tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False)
for seed in [17, 29, 41]:
    data = read(R / f'worlds_{seed}.json'); values = []
    for batch in rows(R / f'schedule_{seed}.jsonl'):
        plain = [render(data[key], pool, n) for key, n in zip(batch['ids'], batch['names'])]
        proof = [render(data[key], pool, n, 'proof')[0] for key, n in zip(batch['ids'], batch['names'])]
        groups = {'query': ([t for t, _ in plain], ['ANSWER:\n' + str(v) for v in batch['labels']]),
                  'actor': ([e['prompt'] for e in batch['actor']], [e['answer'] for e in batch['actor']]),
                  'proof': (proof, ['PROOF: ' + words[0] for _, words in plain])}
        total = {}
        for name, (texts, answers) in groups.items():
            lengths = [length(t) for t in texts]; encoded = [suffix(a) for a in answers]
            assert len({len(s) for s in encoded}) == 1 and all(s[-1] == tokenizer.eos_token_id for s in encoded)
            atoms = [(' ' + words[0]) for _, words in plain] if name == 'proof' else [a[-1] for a in answers]
            assert all(len(tokenizer.encode(a, add_special_tokens=False)) == 1 and s[-2] == tokenizer.encode(a, add_special_tokens=False)[0] for a, s in zip(atoms, encoded))
            total[name + '_tokens'] = sum(n + len(s) for n, s in zip(lengths, encoded)); maximum = max(maximum, max(lengths))
        for _, words in plain:
            assert all(len(tokenizer.encode(' ' + word, add_special_tokens=False)) == 1 for word in words)
        values.append(total)
    totals[str(seed)] = values
obs = rows(R / 'observations.jsonl'); pairs = collections.defaultdict(list); maxima = {}
for mode, key in [('binary', 'prompt'), ('proof', 'proof_prompt')]:
    lengths = [length(o[key]) for o in obs]; maxima[mode] = max(lengths)
    for i, o in enumerate(obs):
        if o['kind'] == 'switch': pairs[mode, o['group'], o['vocab'], o['view']].append((lengths[i], max(lengths[(i // 4) * 4:(i // 4) * 4 + 4])))
assert len(pairs) == 288 and all(len(v) == 2 and len(set(v)) == 1 for v in pairs.values())
assert max(maximum, *maxima.values()) < 1024
result = {'training_tokens_by_step': totals, 'training_max_prompt_tokens': maximum, 'evaluation_max_prompt_tokens': maxima,
          'equal_own_and_padded_pairs': 288, 'target_positions_verified': True, 'model_loaded': False, 'model_calls': 0,
          'script_sha256': sha(R / 'token_preflight.py'), 'plan_sha256': sha(R / 'PLAN.json')}
assert not (R / 'TOKEN_PREFLIGHT.json').exists()
(R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'training_tokens_by_step'}))
