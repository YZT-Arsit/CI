import gc
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from rehearsal_data import R, ROOT, PARENT, read, rows, sha, verify

spec = importlib.util.spec_from_file_location('join_runner_for_rehearsal', PARENT / 'run_join.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner); w = runner.w
assert runner.R == PARENT
evaluate = sys.modules['common'].evaluate


def main():
    plan = verify(); w.verify_model(); preflight = read(R / 'TOKEN_PREFLIGHT.json')
    assert preflight['plan_sha256'] == sha(R / 'PLAN.json') and preflight['script_sha256'] == sha(R / 'token_preflight.py')
    study = R / 'study'; assert not study.exists(); study.mkdir()
    pool = read(R / 'palettes.json')['new']; worlds = read(R / 'dev_worlds.json'); summaries = []; checkpoints = {}
    (study / 'dev_worlds.json').write_bytes((R / 'dev_worlds.json').read_bytes())
    panels = {p: rows(PARENT / f'{p}_observations.jsonl') for p in ['standard', 'factors']}
    panels['oracle'] = rows(R.parent / '20261117_contrast_repair/fresh_observations.jsonl')
    for arm, cfg in plan['arms'].items():
        seed, kind = cfg['seed'], cfg['kind']; cp = plan['checkpoints'][str(seed)]; folder = study / arm; folder.mkdir()
        tokenizer, model = w.load('start', {'checkpoints': {'start': cp}}); initial = w.digest(model)
        assert initial == read(ROOT / cp['path'] / 'EVAL_IDENTITY.json')['before']
        (folder / 'INITIAL_STATE.json').write_text(json.dumps({'digest': initial, 'checkpoint': cp}), encoding='utf-8')
        if kind != 'initial':
            optimizer = w.torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['lr'], weight_decay=0.)
            letters = w.answer_ids(tokenizer)
            for step, batch in enumerate(rows(R / f'training_{seed}.jsonl')):
                def timeout():
                    (folder / 'TIMEOUT.json').write_text(json.dumps({'step': step + 1, 'seconds': 15})); os._exit(124)
                timer = threading.Timer(15, timeout); timer.daemon = True; timer.start(); start = time.perf_counter(); model.train()
                losses = []; tokens = {}
                for branch in ['common', kind]:
                    examples = batch[branch]
                    logits, ce, tokens[branch] = w.forward_targets(tokenizer, model, [e['prompt'] for e in examples], ['ANSWER:\n' + str(e['label']) for e in examples])
                    losses += [ce[:, -2].mean(), .1 * w.torch.cat([ce[:, :-2], ce[:, -1:]], 1).mean()]
                actor = batch['actor']; alogits, ace, tokens['actor'] = w.forward_targets(tokenizer, model, [a['prompt'] for a in actor], [a['answer'] for a in actor])
                losses += list(w.set_loss(alogits, ace, [letters[a['good_indices']] for a in actor])); loss = sum(losses)
                optimizer.zero_grad(); loss.backward(); norm = float(w.torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)); optimizer.step(); timer.cancel()
                assert {int(s['step']) for s in optimizer.state.values() if 'step' in s} == {step + 1}
                w.append(folder / 'training.jsonl', {'step': step, 'optimizer_step': step + 1, 'global_cell': batch['global_cell'],
                    'tokens': tokens, 'losses': [float(v.detach()) for v in losses], 'loss': float(loss.detach()),
                    'gradient_norm': norm, 'seconds': time.perf_counter() - start})
                if (step + 1) % 18 == 0: print(json.dumps({'arm': arm, 'step': step + 1}), flush=True)
                del logits, ce, alogits, ace, loss, losses
            model.save_pretrained(folder / 'adapter'); del optimizer
            (folder / 'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes': {p.name: sha(p) for p in (folder / 'adapter').iterdir() if p.is_file()},
                'steps': plan['steps'], 'initial_adapter_hashes': cp['seal']['adapter_hashes']}, indent=2), encoding='utf-8')
        model.eval(); before = w.digest(model); assert (before == initial) == (kind == 'initial')
        checkpoints[arm] = cp if kind == 'initial' else {'path': folder.relative_to(ROOT).as_posix(), 'seal': read(folder / 'CHECKPOINT_SEALED.json')}
        reused = {}
        for panel, cases in panels.items():
            if kind == 'initial':
                source = ROOT / cp['path'] / (panel + '.jsonl'); shutil.copy2(source, folder / (panel + '.jsonl'))
                reused[panel] = {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source), 'new_calls': 0}; continue
            for start in range(0, len(cases), 4):
                batch = cases[start:start + 4]; texts = [c['prompt'] for c in batch]
                lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': t}], tokenize=True, add_generation_prompt=True)) for t in texts]
                raw, seconds = w.generate(tokenizer, model, texts)
                for case, text, length in zip(batch, raw, lengths): w.append(folder / (panel + '.jsonl'), {'id': case['id'], 'raw': text,
                    'prompt_tokens': length, 'padded_width': max(lengths), 'seconds_batch': seconds})
            print(json.dumps({'arm': arm, 'panel': panel, 'responses': len(cases)}), flush=True)
        if reused: (folder / 'REUSED.json').write_text(json.dumps(reused, indent=2), encoding='utf-8')
        for condition, cases in worlds.items():
            records = evaluate(tokenizer, model, cases, pool, True)
            path = folder / (condition + '.jsonl'); path.write_text(''.join(json.dumps(v) + '\n' for v in records), encoding='utf-8')
            summaries.append({'arm': arm, 'condition': condition, 'episodes': len(records),
                **{k: sum(v[k] for v in records) for k in ['safe_success', 'strict_success', 'invalid', 'invalid_queries']},
                'valid_ci': sum(v['ci'] and not v['invalid'] for v in records), 'raw_sha256': sha(path)})
        after = w.digest(model); assert before == after
        (folder / 'EVAL_IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'checkpoint': checkpoints[arm], 'errors': 0}), encoding='utf-8')
        print(json.dumps({'arm': arm, 'episodes': 32}), flush=True); del model; gc.collect()
    runtime = {**plan, 'arms': list(plan['arms']), 'evidence_arms': list(plan['arms']), 'checkpoint_bindings': checkpoints, 'source_sha256': sha(R / 'run_rehearsal.py')}
    (study / 'PLAN.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    (study / 'RESULTS.json').write_text(json.dumps({'summaries': summaries, 'whole_goal_complete': False}, indent=2), encoding='utf-8')
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in study.rglob('*') if p.is_file()}, 'whole_goal_complete': False}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
