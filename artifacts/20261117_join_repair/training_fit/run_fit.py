import gc
import importlib.util
import json
from join_fit_data import F, R, ROOT, data, read, rows, sha, verified
spec = importlib.util.spec_from_file_location('join_worker_for_fit', R / 'run_join.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner); w = runner.w
assert runner.R == R and w.R == data.PARENT


def main():
    parent, plan = verified(); w.verify_model(); seal = read(R / 'SEALED.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(R / name) == digest for name, digest in seal['files'].items())
    output = F / 'outputs'; assert not output.exists(); output.mkdir(); obs = rows(F / 'observations.jsonl')
    for arm, cfg in plan['arms'].items():
        source = R / 'study' / arm; cp = {'path': source.relative_to(ROOT).as_posix(), 'seal': read(source / 'CHECKPOINT_SEALED.json')}
        tokenizer, model = w.load('start', {'checkpoints': {'start': cp}}); before = w.digest(model)
        assert before == read(source / 'EVAL_IDENTITY.json')['before']
        selected = [o for o in obs if o['seed'] == cfg['seed']]; assert len(selected) == 56
        for start in range(0, 56, 4):
            batch = selected[start:start + 4]; texts = [o['prompt'] for o in batch]
            lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': t}], tokenize=True, add_generation_prompt=True)) for t in texts]
            assert max(lengths) < 1024
            raw, seconds = w.generate(tokenizer, model, texts)
            for o, text, length in zip(batch, raw, lengths):
                w.append(output / (arm + '.jsonl'), {'id': o['id'], 'raw': text, 'prompt_tokens': length, 'padded_width': max(lengths), 'seconds_batch': seconds})
        after = w.digest(model); assert before == after
        (output / (arm + '_IDENTITY.json')).write_text(json.dumps({'before': before, 'after': after, 'checkpoint': cp, 'errors': 0}), encoding='utf-8')
        print(json.dumps({'arm': arm, 'responses': 56}), flush=True); del model; gc.collect()
    (F / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(F / 'PLAN.json'), 'parent_seal_sha256': sha(R / 'SEALED.json'),
        'files': {p.relative_to(F).as_posix(): sha(p) for p in output.iterdir() if p.is_file()}}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
