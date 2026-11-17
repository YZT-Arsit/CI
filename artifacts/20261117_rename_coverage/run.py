"""Matched name-coverage intervention and unfiltered autonomous CPU evaluation."""
import gc
import json
import os
import shutil
import sys
import threading
import time
from rename_data import R, ROOT, PARENT, read, rows, sha, verify
from rename_views import views_for
sys.path[:0] = [str(R.parent / '20261117_frozen_crossover'), str(R.parent / '20261117_teacher_binding')]
from common import load, generate, evaluate, verify as verify_model, torch, palette
from binding_loss import loss_for_views


def append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row) + '\n')


def digest(model):
    import hashlib
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            h.update(name.encode()); h.update(parameter.detach().numpy().tobytes())
    return h.hexdigest()


def main():
    plan = verify(); verify_model(); folder = R / 'study'
    assert not folder.exists(); folder.mkdir()
    shutil.copy2(R / 'PLAN.json', folder / 'PLAN.json')
    shutil.copy2(R / 'dev_worlds.json', folder / 'dev_worlds.json')
    batches = rows(R / 'batches.jsonl'); worlds = read(R / 'dev_worlds.json'); summaries = []
    for arm in plan['arms']:
        target = folder / arm; target.mkdir()
        checkpoint = plan['fixed_checkpoint' if arm == 'fixed' else 'checkpoint']
        tokenizer, model = load('start', {'checkpoints': {'start': checkpoint}})
        pools = read(R / 'palettes.json'); assert palette(tokenizer)['train'] == pools['train']
        (target / 'INITIAL_STATE.json').write_text(json.dumps({'trainable_sha256': digest(model), 'checkpoint': checkpoint['path']}))
        if arm == 'renamed':
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['lr'], weight_decay=0.)
            for step, index in enumerate(plan['schedule']):
                def timeout():
                    (target / 'TIMEOUT.json').write_text(json.dumps({'step': step + 1, 'seconds': 15})); os._exit(124)
                timer = threading.Timer(15, timeout); timer.daemon = True; timer.start()
                start = time.perf_counter(); model.train(); batch = batches[index]
                pair = batch['pairs'][str(batch['mixed_depth'])]; views = views_for(batch, step)
                loss, parts, tokens, predicted, supplied = loss_for_views(tokenizer, model, pair, pools['train'], views, step, 0., teacher=True)
                opt.zero_grad(); loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)); opt.step(); timer.cancel()
                assert {int(s['step']) for s in opt.state.values() if 'step' in s} == {step + 1}
                append(target / 'training.jsonl', {'step': step + 1, 'batch_index': index, 'loss': float(loss.detach()),
                       'query_loss': float(parts['query_loss'].detach()), 'actor_loss': float(parts['actor_loss'].detach()),
                       'seconds': time.perf_counter() - start, 'tokens': tokens, 'gradient_norm': norm,
                       'optimizer_step': step + 1, 'supplied': supplied, 'views': views})
                if (step + 1) % 12 == 0: print(json.dumps({'arm': arm, 'step': step + 1}), flush=True)
                del loss, parts
            model.save_pretrained(target / 'adapter')
            (target / 'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes': {p.name: sha(p) for p in (target / 'adapter').iterdir() if p.is_file()},
                'steps': 144, 'initial_adapter_hashes': plan['checkpoint']['seal']['adapter_hashes']}, indent=2))
            del opt
        else:
            source = ROOT / checkpoint['path']
            shutil.copytree(source / 'adapter', target / 'adapter')
            shutil.copy2(source / 'CHECKPOINT_SEALED.json', target / 'CHECKPOINT_SEALED.json')
        model.eval(); before = digest(model)
        for kind, source in [('query', R / 'observations.jsonl'), ('oracle', R.parent / '20261117_contrast_repair/fresh_observations.jsonl'),
                             ('third', R / 'third_observations.jsonl'), ('dev', R / 'dev_observations.jsonl')]:
            if arm == 'fixed' and kind in ['query', 'oracle']:
                original = PARENT / 'fit/joint' / (kind + '_raw.jsonl')
                shutil.copy2(original, target / (kind + '_raw.jsonl'))
                (target / ('REUSED_' + kind.upper() + '.json')).write_text(json.dumps({'source': original.relative_to(ROOT).as_posix(), 'sha256': sha(original), 'new_calls': 0}))
                continue
            observations = rows(source)
            for start in range(0, len(observations), 4):
                batch = observations[start:start + 4]
                lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': c['prompt']}], tokenize=True, add_generation_prompt=True)) for c in batch]
                raw, seconds = generate(tokenizer, model, [c['prompt'] for c in batch])
                for case, value, length in zip(batch, raw, lengths):
                    append(target / (kind + '_raw.jsonl'), {'id': case['id'], 'raw': value, 'seconds_batch': seconds, 'prompt_tokens': length})
            print(json.dumps({'arm': arm, 'panel': kind, 'responses': len(observations)}), flush=True)
        for condition, ws in worlds.items():
            result = evaluate(tokenizer, model, ws, pools['new'], True)
            path = target / (condition + '.jsonl'); path.write_text(''.join(json.dumps(r) + '\n' for r in result))
            summary = {'arm': arm, 'condition': condition, 'episodes': len(result),
                       **{k: sum(r[k] for r in result) for k in ['safe_success', 'strict_success', 'invalid', 'invalid_queries']},
                       'valid_ci': sum(r['ci'] and not r['invalid'] for r in result), 'raw_sha256': sha(path)}
            summaries.append(summary); print(json.dumps(summary), flush=True)
        after = digest(model); assert before == after
        (target / 'EVAL_IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'errors': 0}))
        del model; gc.collect()
    (folder / 'RESULTS.json').write_text(json.dumps({'summaries': summaries, 'whole_goal_complete': False}, indent=2))
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()}, 'whole_goal_complete': False}, indent=2))


if __name__ == '__main__':
    main()
