import gc
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from join_data import R, ROOT, PARENT, prior, read, rows, sha, verify
sys.path.insert(0, str(PARENT))
spec = importlib.util.spec_from_file_location('witness_worker_for_join', PARENT / 'run.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w); assert w.R == PARENT


def main():
    plan = verify(); w.verify_model(); token = read(R / 'TOKEN_PREFLIGHT.json')
    assert token['plan_sha256'] == sha(R / 'PLAN.json') and token['script_sha256'] == sha(R / 'token_preflight.py')
    study = R / 'study'; assert not study.exists(); study.mkdir(); pool = read(prior.NAMES / 'palettes.json')['train']
    oracle = rows(R.parent / '20261117_contrast_repair/fresh_observations.jsonl')
    for arm, cfg in plan['arms'].items():
        seed, kind = cfg['seed'], cfg['kind']; cp = plan['checkpoints'][str(seed)]; folder = study / arm; folder.mkdir()
        tokenizer, model = w.load('start', {'checkpoints': {'start': cp}}); initial = w.digest(model)
        (folder / 'INITIAL_STATE.json').write_text(json.dumps({'digest': initial, 'checkpoint': cp}), encoding='utf-8')
        if kind != 'initial':
            worlds = read(R / f'worlds_{seed}.json')[kind]; schedule = rows(R / f'schedule_{seed}.jsonl')
            optimizer = w.torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['lr'], weight_decay=0.)
            letters = w.answer_ids(tokenizer)
            for step, batch in enumerate(schedule):
                def timeout():
                    (folder / 'TIMEOUT.json').write_text(json.dumps({'step': step + 1, 'seconds': 15})); os._exit(124)
                timer = threading.Timer(15, timeout); timer.daemon = True; timer.start(); start = time.perf_counter(); model.train()
                texts = [prior.render(worlds[key], pool, name)[0] for key, name in zip(batch['ids'], batch['names'])]
                logits, ce, qt = w.forward_targets(tokenizer, model, texts, ['ANSWER:\n' + str(v) for v in batch['labels']])
                bit = ce[:, -2].mean(); formatting = .1 * w.torch.cat([ce[:, :-2], ce[:, -1:]], 1).mean()
                actor = batch['actor']; alogits, ace, at = w.forward_targets(tokenizer, model, [a['prompt'] for a in actor], [a['answer'] for a in actor])
                abit, af = w.set_loss(alogits, ace, [letters[a['good_indices']] for a in actor]); loss = bit + formatting + abit + af
                optimizer.zero_grad(); loss.backward(); norm = float(w.torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)); optimizer.step(); timer.cancel()
                assert {int(s['step']) for s in optimizer.state.values() if 'step' in s} == {step + 1}
                w.append(folder / 'training.jsonl', {k: batch[k] for k in ['step', 'stratum', 'ids', 'labels', 'names']} |
                         {'actor_ids': [a['id'] for a in actor], 'optimizer_step': step + 1, 'loss': float(loss.detach()),
                          'bit_loss': float(bit.detach()), 'format_loss': float(formatting.detach()), 'actor_bit_loss': float(abit.detach()),
                          'actor_format_loss': float(af.detach()), 'query_tokens': qt, 'actor_tokens': at,
                          'gradient_norm': norm, 'seconds': time.perf_counter() - start})
                if (step + 1) % 24 == 0: print(json.dumps({'arm': arm, 'step': step + 1}), flush=True)
                del logits, ce, alogits, ace, bit, formatting, abit, af, loss
            model.save_pretrained(folder / 'adapter'); del optimizer
            (folder / 'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes': {p.name: sha(p) for p in (folder / 'adapter').iterdir() if p.is_file()},
                'steps': 96, 'initial_adapter_hashes': cp['seal']['adapter_hashes']}, indent=2), encoding='utf-8')
        model.eval(); before = w.digest(model); assert kind == 'initial' or before != initial
        for panel in ['standard', 'factors', 'oracle']:
            if panel == 'oracle' and kind == 'initial':
                source = ROOT / cp['path'] / 'oracle.jsonl'; shutil.copy2(source, folder / 'oracle.jsonl')
                (folder / 'REUSED_ORACLE.json').write_text(json.dumps({'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source), 'new_calls': 0})); continue
            cases = oracle if panel == 'oracle' else rows(R / f'{panel}_observations.jsonl')
            for start in range(0, len(cases), 4):
                batch = cases[start:start + 4]; texts = [c['prompt'] for c in batch]
                lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': t}], tokenize=True, add_generation_prompt=True)) for t in texts]
                raw, seconds = w.generate(tokenizer, model, texts)
                for case, text, length in zip(batch, raw, lengths): w.append(folder / (panel + '.jsonl'), {'id': case['id'], 'raw': text,
                    'prompt_tokens': length, 'padded_width': max(lengths), 'seconds_batch': seconds})
            print(json.dumps({'arm': arm, 'panel': panel, 'responses': len(cases)}), flush=True)
        after = w.digest(model); assert before == after
        (folder / 'EVAL_IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'errors': 0})); del model; gc.collect()
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in study.rglob('*') if p.is_file()}, 'whole_goal_complete': False}, indent=2))


if __name__ == '__main__': main()
