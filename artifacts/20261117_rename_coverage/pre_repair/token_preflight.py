"""Tokenizer-only verification; no model loaded and no generation."""
import json
import random
import sys
from rename_data import R, PARENT, read, rows, sha, verify, batches, schedule
from rename_views import views_for
sys.path.insert(0, str(R.parent / '20261117_teacher_binding'))
from binding_loss import response
from render import query, actor
from lm_common import AutoTokenizer, env


def main():
    verify(); tokenizer = AutoTokenizer.from_pretrained(env()['native_lm_path'], local_files_only=True)
    pool = read(R / 'palettes.json')['train']; bs = batches(); old = rows(PARENT / 'fit/joint/training.jsonl'); totals = []
    for step, index in enumerate(schedule()):
        b = bs[index]; pair = b['pairs'][str(b['mixed_depth'])]; count = 0
        for seeds in views_for(b, step):
            for side, row in enumerate(pair):
                examples = [(query(row['observation'], pool, seeds, c['action']), 'ANSWER:\n' + str(c['safe_completion'])) for c in row['candidates']]
                text, order = actor(row['observation'], pool, seeds, {c['action']: str(c['safe_completion']) for c in row['candidates']})
                answer, good = response(row, order, pool, seeds['names'], 'plain', random.Random(step * 2 + side))
                examples.append((text, answer))
                for text, answer in examples:
                    count += len(tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True))
                    count += len(tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False))
        totals.append(count)
    match = totals == [v['tokens'] for v in old]
    report = {'per_update_tokens': totals, 'matches_all_fixed_updates': match, 'model_loaded': False, 'model_calls': 0,
              'script_sha256': sha(R / 'token_preflight.py'), 'plan_sha256': sha(R / 'PLAN.json')}
    assert not (R / 'TOKEN_PREFLIGHT.json').exists()
    (R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_update_tokens'}), flush=True)
    assert match


if __name__ == '__main__':
    main()
