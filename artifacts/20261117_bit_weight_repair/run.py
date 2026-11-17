"""One matched joint-loss continuation; baseline evidence is not regenerated."""
import gc
import json
import os
import sys
import threading
import time
from repair_data import R, ROOT, PARENT, read, rows, sha, verify

sys.path.insert(0, str(R.parent / '20261117_frozen_crossover'))
from common import load, generate, verify as verify_model, torch, palette
from fixed_loss import loss_for_views


def digest(model):
    import hashlib
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode()); h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def append(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value) + '\n')


def main():
    plan = verify(); verify_model()
    folder = R / 'fit'; assert not folder.exists(); folder.mkdir()
    tokenizer, model = load('start', {'checkpoints': {'start': plan['checkpoint']}})
    pools = palette(tokenizer); batches = rows(PARENT / 'batches.jsonl')
    (folder / 'INITIAL_STATE.json').write_text(json.dumps({'trainable_sha256': digest(model)}))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['lr'], weight_decay=0.)
    for step, index in enumerate(plan['schedule']):
        def timeout():
            (folder / 'TIMEOUT.json').write_text(json.dumps({'step': step + 1, 'seconds': 15})); os._exit(124)
        timer = threading.Timer(15, timeout); timer.daemon = True; timer.start()
        start = time.perf_counter(); model.train(); batch = batches[index]
        total, parts, tokens, predicted, supplied = loss_for_views(tokenizer, model, batch['pairs'][str(batch['mixed_depth'])], pools['train'], batch['views'], step, 0., teacher=True)
        opt.zero_grad(); total.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)); opt.step(); timer.cancel()
        assert {int(s['step']) for s in opt.state.values() if 'step' in s} == {step + 1}
        append(folder / 'training.jsonl', {'step': step + 1, 'batch_index': index, 'optimizer_step': step + 1,
               'loss': float(total.detach()), 'query_loss': float(parts['query_loss'].detach()), 'actor_loss': float(parts['actor_loss'].detach()),
               'seconds': time.perf_counter() - start, 'tokens': tokens, 'gradient_norm': norm, 'supplied': supplied})
        del total, parts
        if (step + 1) % 24 == 0: print(json.dumps({'step': step + 1}), flush=True)
    model.save_pretrained(folder / 'adapter')
    (folder / 'CHECKPOINT_SEALED.json').write_text(json.dumps({'steps': 144, 'initial_adapter_hashes': plan['checkpoint']['seal']['adapter_hashes'],
        'adapter_hashes': {p.name: sha(p) for p in (folder / 'adapter').iterdir() if p.is_file()}}, indent=2))
    del opt; model.eval(); before = digest(model)
    oracle_source = R.parent / '20261117_contrast_repair/fresh_observations.jsonl'
    for kind, source in [('query', PARENT / 'observations.jsonl'), ('oracle', oracle_source)]:
        observations = rows(source)
        for start in range(0, len(observations), 4):
            batch = observations[start:start + 4]
            lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': c['prompt']}], tokenize=True, add_generation_prompt=True)) for c in batch]
            values, seconds = generate(tokenizer, model, [c['prompt'] for c in batch])
            for case, value, length in zip(batch, values, lengths):
                append(folder / (kind + '_raw.jsonl'), {'id': case['id'], 'raw': value, 'seconds_batch': seconds, 'prompt_tokens': length})
    after = digest(model); assert before == after
    (folder / 'EVAL_IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'errors': 0}))
    (folder / 'CONTROL_RECEIPT.json').write_text(json.dumps({'source': (PARENT / 'fit/joint').relative_to(ROOT).as_posix(),
        'source_seal_sha256': sha(PARENT / 'SEALED.json'), 'source_results_sha256': sha(PARENT / 'RESULTS.json'), 'new_control_calls': 0}, indent=2))
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()}, 'whole_goal_complete': False}, indent=2))
    del model; gc.collect()
    print(json.dumps({'complete': True, 'new_responses': 480}), flush=True)


if __name__ == '__main__':
    main()
