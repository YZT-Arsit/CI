"""Frozen-model inference on matched components; no training or fallback."""
import gc
import importlib.util
import json
import sys
from factor_data import R, ROOT, PARENT, read, rows, sha, verify
sys.path.insert(0, str(PARENT))
spec = importlib.util.spec_from_file_location('witness_worker_for_factors', PARENT / 'run.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
assert worker.R == PARENT


def main():
    plan = verify(); worker.verify_model(); token = read(R / 'TOKEN_PREFLIGHT.json')
    assert token['plan_sha256'] == sha(R / 'PLAN.json') and token['script_sha256'] == sha(R / 'token_preflight.py')
    output = R / 'outputs'; assert not output.exists(); output.mkdir()
    for arm, cp in plan['models'].items():
        tokenizer, model = worker.load('start', {'checkpoints': {'start': cp}}); before = worker.digest(model)
        assert before == read(ROOT / cp['path'] / 'EVAL_IDENTITY.json')['before']
        for mode in cp['modes']:
            obs = rows(R / ('echo_observations.jsonl' if mode == 'echo' else 'observations.jsonl'))
            for start in range(0, len(obs), 4):
                batch = obs[start:start + 4]; key = 'proof_prompt' if mode == 'proof' else 'prompt'; texts = [o[key] for o in batch]
                lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': t}], tokenize=True, add_generation_prompt=True)) for t in texts]
                raw, elapsed = worker.generate(tokenizer, model, texts)
                for o, text, length in zip(batch, raw, lengths):
                    worker.append(output / f'{arm}_{mode}.jsonl', {'id': o['id'], 'raw': text, 'seconds_batch': elapsed,
                                                                 'prompt_tokens': length, 'padded_width': max(lengths)})
            print(json.dumps({'arm': arm, 'mode': mode, 'responses': len(obs)}), flush=True)
        after = worker.digest(model); assert before == after
        (output / f'{arm}_IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'checkpoint': cp, 'errors': 0}), encoding='utf-8')
        del model; gc.collect()
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in output.iterdir() if p.is_file()}, 'whole_goal_complete': False}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
