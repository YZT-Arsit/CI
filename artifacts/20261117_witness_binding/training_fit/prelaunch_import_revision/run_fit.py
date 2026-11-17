"""Generate selected TRAIN answers from the nine completed frozen models."""
import gc
import json
from data import F, R, ROOT, read, rows, sha, verified
from run import digest, append, load, generate, verify_model


def main():
    main_plan, plan = verified(); verify_model(); parent = read(R / 'SEALED.json')
    assert parent['complete'] and all(sha(R / n) == h for n, h in parent['files'].items())
    output = F / 'outputs'; assert not output.exists(); output.mkdir()
    obs = rows(F / 'observations.jsonl')
    for arm, config in main_plan['arms'].items():
        source = R / 'study' / arm; cp = {'path': source.relative_to(ROOT).as_posix(), 'seal': read(source / 'CHECKPOINT_SEALED.json')}
        tokenizer, model = load('start', {'checkpoints': {'start': cp}}); before = digest(model)
        assert before == read(source / 'EVAL_IDENTITY.json')['before']
        selected = [o for o in obs if o['seed'] == config['seed']]; assert len(selected) == 48
        for mode in ['binary', 'proof']:
            for start in range(0, 48, 4):
                batch = selected[start:start + 4]; key = 'proof_prompt' if mode == 'proof' else 'prompt'; texts = [o[key] for o in batch]
                lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': t}], tokenize=True, add_generation_prompt=True)) for t in texts]
                raw, elapsed = generate(tokenizer, model, texts)
                for o, text, length in zip(batch, raw, lengths):
                    append(output / f'{arm}_{mode}.jsonl', {'id': o['id'], 'raw': text, 'seconds_batch': elapsed,
                                                          'prompt_tokens': length, 'padded_width': max(lengths)})
        after = digest(model); assert before == after
        (output / f'{arm}_IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'checkpoint': cp, 'errors': 0}), encoding='utf-8')
        print(json.dumps({'arm': arm, 'responses': 96}), flush=True); del model; gc.collect()
    (F / 'SEALED.json').write_text(json.dumps({'complete': True, 'parent_seal_sha256': sha(R / 'SEALED.json'),
        'plan_sha256': sha(F / 'PLAN.json'), 'files': {p.relative_to(F).as_posix(): sha(p) for p in output.iterdir() if p.is_file()}}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
