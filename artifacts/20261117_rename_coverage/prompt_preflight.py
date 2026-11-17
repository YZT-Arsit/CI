"""Individual prompt/answer length checks, beyond aggregate token equality."""
from token_preflight import R, read, sha, verify, batches, schedule, views_for, query, actor, response, random, AutoTokenizer, env, json
from rename_data import cases, third_cases
from collections import defaultdict


def main():
    verify(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
    pool = read(R / 'palettes.json')['train']; bs = batches(); comparisons = 0; mismatches = []
    def lengths(text, answer):
        return [len(tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True)),
                len(tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False))]
    for step, index in enumerate(schedule()):
        b = bs[index]; pair = b['pairs'][str(b['mixed_depth'])]
        for vi, (before, after) in enumerate(zip(b['views'], views_for(b, step))):
            for side, row in enumerate(pair):
                rendered = []
                for seeds in [before, after]:
                    examples = [(query(row['observation'], pool, seeds, c['action']), 'ANSWER:\n' + str(c['safe_completion'])) for c in row['candidates']]
                    text, order = actor(row['observation'], pool, seeds, {c['action']: str(c['safe_completion']) for c in row['candidates']})
                    answer, good = response(row, order, pool, seeds['names'], 'plain', random.Random(step * 2 + side))
                    examples.append((text, answer)); rendered.append([lengths(*example) for example in examples])
                for ci, (a, z) in enumerate(zip(*rendered)):
                    comparisons += 1
                    if a != z: mismatches.append({'step': step, 'view': vi, 'side': side, 'example': ci, 'before': a, 'after': z})
    groups = defaultdict(list)
    for o in cases()[0] + third_cases()[0]: groups[o['group']].append(lengths(o['prompt'], '')[0])
    token_control = len(groups) == 84 and all(len(v) == 6 and len(set(v)) == 1 for v in groups.values())
    report = {'individual_training_comparisons': comparisons, 'mismatches': mismatches, 'six_prompt_groups': len(groups),
              'six_prompt_length_control': token_control, 'model_loaded': False, 'model_calls': 0,
              'script_sha256': sha(R / 'prompt_preflight.py'), 'plan_sha256': sha(R / 'PLAN.json')}
    assert not (R / 'PROMPT_PREFLIGHT.json').exists()
    (R / 'PROMPT_PREFLIGHT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True); assert not mismatches and token_control


if __name__ == '__main__':
    main()
