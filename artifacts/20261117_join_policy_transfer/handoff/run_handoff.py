import gc
import importlib.util
import json
from join_handoff_data import H, R, read, rows, sha, verify
spec = importlib.util.spec_from_file_location('policy_runner_for_handoff', R / 'run_policy.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner); w = runner.w
assert runner.R == R


def main():
    plan = verify(); w.verify_model(); token = read(H / 'TOKEN_PREFLIGHT.json')
    assert token['plan_sha256'] == sha(H / 'PLAN.json') and token['script_sha256'] == sha(H / 'token_preflight.py')
    output = H / 'outputs'; assert not output.exists(); output.mkdir(); obs = rows(H / 'observations.jsonl')
    cps = read(R / 'study/PLAN.json')['checkpoint_bindings']
    for arm in plan['arms']:
        tokenizer, model = w.load('start', {'checkpoints': {'start': cps[arm]}}); before = w.digest(model)
        assert before == read(R / 'study' / arm / 'EVAL_IDENTITY.json')['before']
        for start in range(0, 32, 4):
            batch = obs[start:start + 4]; texts = [o['prompt'] for o in batch]
            lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': t}], tokenize=True, add_generation_prompt=True)) for t in texts]
            raw, elapsed = w.generate(tokenizer, model, texts)
            for o, text, length in zip(batch, raw, lengths):
                w.append(output / (arm + '.jsonl'), {'id': o['id'], 'raw': text, 'prompt_tokens': length, 'padded_width': max(lengths), 'seconds_batch': elapsed})
        after = w.digest(model); assert before == after
        (output / (arm + '_IDENTITY.json')).write_text(json.dumps({'before': before, 'after': after, 'checkpoint': cps[arm], 'errors': 0}), encoding='utf-8')
        print(json.dumps({'arm': arm, 'responses': 32}), flush=True); del model; gc.collect()
    (H / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(H / 'PLAN.json'), 'parent_seal_sha256': sha(R / 'SEALED.json'),
        'files': {p.relative_to(H).as_posix(): sha(p) for p in output.iterdir() if p.is_file()}}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
