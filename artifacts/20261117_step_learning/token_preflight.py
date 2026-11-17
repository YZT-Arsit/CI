"""Check every scheduled training prompt and confirmation pair without a model."""
import collections
import json
import sys
from learning_data import R, NAMES, read, rows, sha, verify, prompt
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env


def main():
    plan = verify(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
    pool = read(NAMES / 'palettes.json')['train']; data = read(R / 'training_worlds.json'); totals = {}; training_max = 0
    def length(text):
        return len(tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True))
    for bit in [0, 1]:
        answer = tokenizer.encode('ANSWER:\n' + str(bit) + tokenizer.eos_token, add_special_tokens=False)
        assert answer[-2] == tokenizer.encode(str(bit), add_special_tokens=False)[0] and answer[-1] == tokenizer.eos_token_id
    for seed in plan['training_seeds']:
        totals[str(seed)] = []
        for record in rows(R / f'schedule_{seed}.jsonl'):
            lengths = [length(prompt(data[key], pool, name)) for key, name in zip(record['ids'], record['names'])]
            suffixes = [len(tokenizer.encode('ANSWER:\n' + str(bit) + tokenizer.eos_token, add_special_tokens=False)) for bit in record['labels']]
            totals[str(seed)].append(sum(a + b for a, b in zip(lengths, suffixes))); training_max = max(training_max, max(lengths))
            if record['kind'] == 'switch': assert lengths[0] == lengths[1] and lengths[2] == lengths[3]
    observations = rows(R / 'observations.jsonl'); groups = collections.defaultdict(list); maximum = {}
    for mode in ['zero_shot', 'four_examples']:
        obs = [o for o in observations if o['mode'] == mode]; lengths = [length(o['prompt']) for o in obs]; maximum[mode] = max(lengths)
        for i, o in enumerate(obs):
            if o['kind'] == 'switch': groups[mode, o['group'], o['vocab'], o['view']].append((lengths[i], max(lengths[(i // 4) * 4:(i // 4) * 4 + 4])))
    assert len(groups) == 256 and all(len(v) == 2 and len(set(v)) == 1 for v in groups.values())
    assert max(training_max, *maximum.values()) < 1024
    result = {'training_max_prompt_tokens': training_max, 'confirmation_max_prompt_tokens': maximum,
              'training_tokens_by_step': totals, 'paired_groups_equal_own_and_padded_lengths': 256,
              'answer_bit_position_verified': True, 'model_loaded': False, 'model_calls': 0,
              'script_sha256': sha(R / 'token_preflight.py'), 'plan_sha256': sha(R / 'PLAN.json')}
    assert not (R / 'TOKEN_PREFLIGHT.json').exists(); (R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'training_tokens_by_step'}), flush=True)


if __name__ == '__main__': main()
