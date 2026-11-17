import gc
import importlib.util
import json
import re
from bridge_data import R, ROOT, PARENT, read, rows, sha, verify
spec = importlib.util.spec_from_file_location('rehearsal_runner_for_fact_bridge', PARENT / 'run_rehearsal.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner); w = runner.w
assert runner.R == PARENT


def main():
    plan = verify(); w.verify_model(); token = read(R / 'TOKEN_PREFLIGHT.json')
    assert token['plan_sha256'] == sha(R / 'PLAN.json') and token['script_sha256'] == sha(R / 'token_preflight.py')
    states, queries = rows(R / 'states.jsonl'), rows(R / 'queries.jsonl'); study = R / 'study'; assert not study.exists(); study.mkdir()
    for index, seed in enumerate(plan['seeds']):
        cp = plan['checkpoints'][str(seed)]; folder = study / str(seed); folder.mkdir()
        tokenizer, model = w.load('start', {'checkpoints': {'start': cp}}); before = w.digest(model)
        assert before == read(ROOT / cp['path'] / 'EVAL_IDENTITY.json')['before']
        order = plan['modes'][index:] + plan['modes'][:index]
        for mode in order:
            estimates = {s['id']: {} for s in states}
            for start in range(0, len(queries), 4):
                batch = queries[start:start + 4]; raw, seconds = w.generate(tokenizer, model, [q['prompts'][mode] for q in batch])
                for q, text in zip(batch, raw):
                    found = re.findall(r'ANSWER:\s*([01])\b', text); valid = len(found) == 1 and seconds < 9.5
                    estimate = found[0] if valid else 'unknown'; estimates[q['state_id']][q['action']] = estimate
                    w.append(folder / (mode + '_queries.jsonl'), {'id': q['id'], 'raw': text, 'estimate': estimate, 'valid': valid, 'seconds_batch': seconds})
            for start in range(0, len(states), 4):
                batch = states[start:start + 4]; texts = []
                for s in batch:
                    text = s['actor_prefix'] + 'Own estimates (1 means compliant completion remains possible): '
                    text += '; '.join(chr(65 + i) + '=' + estimates[s['id']][a] for i, a in enumerate(s['actions'])) + '.\n'
                    texts.append(text + '\nReply ANSWER:\nletter.')
                raw, seconds = w.generate(tokenizer, model, texts)
                for s, prompt, answer in zip(batch, texts, raw):
                    found = re.findall(r'ANSWER:\s*([ABC])\b', answer)
                    valid = len(found) == 1 and ord(found[0]) - 65 < len(s['actions']) and seconds < 9.5
                    selected = s['actions'][ord(found[0]) - 65] if valid else None
                    w.append(folder / (mode + '_actions.jsonl'), {'id': s['id'], 'prompt': prompt, 'estimates': estimates[s['id']],
                        'raw': answer, 'valid': valid, 'selected': selected, 'seconds_batch': seconds})
            assert w.digest(model) == before
            print(json.dumps({'seed': seed, 'mode': mode, 'queries': 192, 'actions': 80}), flush=True)
        (folder / 'IDENTITY.json').write_text(json.dumps({'before': before, 'after': w.digest(model), 'checkpoint': cp, 'mode_order': order}), encoding='utf-8')
        del model; gc.collect()
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in study.rglob('*') if p.is_file()}}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
