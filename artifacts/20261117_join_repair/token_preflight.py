import collections
import json
import sys
from join_data import R, prior, read, rows, sha, verify
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env
plan = verify(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
pool = read(prior.NAMES / 'palettes.json')['train']; training = {}; maximum = 0
def length(t): return len(tokenizer.apply_chat_template([{'role': 'user', 'content': t}], tokenize=True, add_generation_prompt=True))
def targets(a): return tokenizer.encode(a + tokenizer.eos_token, add_special_tokens=False)
for seed in [17, 29, 41]:
    data = read(R / f'worlds_{seed}.json'); training[str(seed)] = []
    for batch in rows(R / f'schedule_{seed}.jsonl'):
        text = {k: [prior.render(data[k][key], pool, n)[0] for key, n in zip(batch['ids'], batch['names'])] for k in data}
        ls = {k: [length(t) for t in ts] for k, ts in text.items()}; assert ls['random'] == ls['targeted']
        answers = ['ANSWER:\n' + str(v) for v in batch['labels']]; actor_answers = [e['answer'] for e in batch['actor']]
        for answer in answers + actor_answers:
            ids = targets(answer); atom = tokenizer.encode(answer[-1], add_special_tokens=False)
            assert len(atom) == 1 and ids[-2] == atom[0] and ids[-1] == tokenizer.eos_token_id
        actor_lengths = [length(e['prompt']) for e in batch['actor']]
        maximum = max(maximum, *ls['random'], *actor_lengths)
        training[str(seed)].append({'query_tokens': sum(n + len(targets(a)) for n, a in zip(ls['random'], answers)),
                                   'actor_tokens': sum(n + len(targets(a)) for n, a in zip(actor_lengths, actor_answers))})
lengths = {}; pairs = collections.defaultdict(list); clutter = collections.defaultdict(list)
for panel in ['standard', 'factors']:
    obs = rows(R / f'{panel}_observations.jsonl'); lengths[panel] = [length(o['prompt']) for o in obs]
    for i, o in enumerate(obs):
        value = (lengths[panel][i], max(lengths[panel][i // 4 * 4:i // 4 * 4 + 4]))
        if panel == 'standard' and o['kind'] == 'switch': key = (panel, o['group'], o['vocab'], o['view']); pairs[key].append(value)
        elif panel == 'factors':
            pairs[panel, o['seed'], o['known'], o['noise'], o['vocab']].append(value)
            if o['noise'] != 'none': clutter[o['seed'], o['known'], o['vocab'], o['side']].append(value)
assert len(pairs) == 288 and all(len(v) == 2 and len(set(v)) == 1 for v in pairs.values())
assert len(clutter) == 96 and all(len(v) == 2 and len(set(v)) == 1 for v in clutter.values())
assert max(maximum, *(max(v) for v in lengths.values())) < 1024
result = {'training_tokens': training, 'eval_lengths': lengths, 'matched_training_batches': 288,
          'equal_pairs': 288, 'equal_count_clutter_groups': 96, 'training_max_tokens': maximum,
          'eval_max_tokens': {p: max(v) for p, v in lengths.items()}, 'model_loaded': False, 'model_calls': 0,
          'plan_sha256': sha(R / 'PLAN.json'), 'script_sha256': sha(R / 'token_preflight.py')}
assert not (R / 'TOKEN_PREFLIGHT.json').exists(); (R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ['training_tokens', 'eval_lengths']}))
