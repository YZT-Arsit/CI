import json
import sys
from rehearsal_data import R, PARENT, read, rows, sha, verify, targets
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env
plan = verify(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
def length(t): return len(tokenizer.apply_chat_template([{'role': 'user', 'content': t}], tokenize=True, add_generation_prompt=True))
def suffix(a):
    ids = tokenizer.encode(a + tokenizer.eos_token, add_special_tokens=False); atom = tokenizer.encode(a[-1], add_special_tokens=False)
    assert len(atom) == 1 and ids[-2] == atom[0] and ids[-1] == tokenizer.eos_token_id
    return len(ids)
training = {}; maximum = 0
for seed in [17, 29, 41]:
    training[str(seed)] = []
    for b in rows(R / f'training_{seed}.jsonl'):
        token_counts = {}
        for branch in ['common', 'local', 'global', 'actor']:
            sizes = [length(e['prompt']) + suffix(e['answer'] if branch == 'actor' else 'ANSWER:\n' + str(e['label'])) for e in b[branch]]
            token_counts[branch] = sum(sizes); maximum = max(maximum, *sizes)
        training[str(seed)].append(token_counts)
eval_lengths = {p: [length(o['prompt']) for o in rows(PARENT / f'{p}_observations.jsonl')] for p in ['standard', 'factors']}
oracle_max = max(length(o['prompt']) for o in rows(R.parent / '20261117_contrast_repair/fresh_observations.jsonl'))
from kci.relation_language import prompt, seed_for
policy_lengths = []; pool = read(R / 'palettes.json')['new']
for group in read(R / 'dev_worlds.json').values():
    for world in group:
        for case in targets(world):
            obs = case['observation']; phase = ['Planner', 'Source', 'Worker'].index(obs['role']); name = seed_for(world['base_group_id'], phase)
            text, actions = prompt(obs, pool, name)
            # UNKNOWN is the longest supported estimate; this checks all reachable states.
            text += 'Own estimates (1 means compliant completion remains possible): ' + '; '.join(chr(65 + i) + '=unknown' for i in range(len(actions))) + '.\n\nReply ANSWER:\nletter.'
            policy_lengths.append(length(text))
            for action in obs['actions']:
                q, _ = prompt(obs, pool, name, question='task', candidate=action)
                policy_lengths.append(length(q.replace('A yes; B no.', 'Reply ANSWER: followed by 1 if possible, otherwise 0.')))
assert max(oracle_max, *policy_lengths) < 1024
assert max(maximum, *(max(v) for v in eval_lengths.values())) < 1024
result = {'training_tokens': training, 'eval_lengths': eval_lengths, 'max_training_tokens': maximum, 'oracle_max_tokens': oracle_max,
          'native_policy_max_tokens': max(policy_lengths), 'model_calls': 0,
          'plan_sha256': sha(R / 'PLAN.json'), 'script_sha256': sha(R / 'token_preflight.py')}
assert not (R / 'TOKEN_PREFLIGHT.json').exists(); (R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ['training_tokens', 'eval_lengths']}))
