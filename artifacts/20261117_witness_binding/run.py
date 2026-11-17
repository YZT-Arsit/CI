"""Matched binary versus extra copied-witness supervision on shared LoRA."""
import gc
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from graph_data import R, ROOT, PARENT, NAMES, read, rows, sha, verify, render
sys.path[:0] = [str(R.parent / '20261117_frozen_crossover'), str(R.parent / '20261117_teacher_binding')]
from common import load, generate, verify as verify_model, torch
from binding_loss import forward_targets, answer_ids


def append(path, row):
    with path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row) + '\n')


def digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad: h.update(name.encode()); h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def set_loss(logits, ce, allowed):
    bit = torch.stack([torch.logsumexp(logits[i, -2], 0) - torch.logsumexp(logits[i, -2, ids], 0) for i, ids in enumerate(allowed)]).mean()
    formatting = .1 * torch.cat([ce[:, :-2], ce[:, -1:]], 1).mean()
    return bit, formatting


def main():
    plan = verify(); verify_model(); folder = R / 'study'; assert not folder.exists(); folder.mkdir()
    pool = read(NAMES / 'palettes.json')['train']; obs = rows(R / 'observations.jsonl')
    oracle = rows(R.parent / '20261117_contrast_repair/fresh_observations.jsonl')
    for arm, config in plan['arms'].items():
        seed, kind = config['seed'], config['kind']; cp = plan['checkpoints'][str(seed)]; target = folder / arm; target.mkdir()
        tokenizer, model = load('start', {'checkpoints': {'start': cp}})
        (target / 'INITIAL_STATE.json').write_text(json.dumps({'trainable_sha256': digest(model), 'checkpoint': cp['path']}))
        if kind != 'start':
            data = read(R / f'worlds_{seed}.json'); schedule = rows(R / f'schedule_{seed}.jsonl')
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['lr'], weight_decay=0.)
            letters = answer_ids(tokenizer)
            for step, batch in enumerate(schedule):
                def timeout():
                    (target / 'TIMEOUT.json').write_text(json.dumps({'step': step + 1, 'seconds': 15})); os._exit(124)
                timer = threading.Timer(15, timeout); timer.daemon = True; timer.start(); start = time.perf_counter(); model.train()
                rendered = [render(data[key], pool, name) for key, name in zip(batch['ids'], batch['names'])]
                texts = [t for t, _ in rendered]; valid = [v for _, v in rendered]
                logits, ce, qt = forward_targets(tokenizer, model, texts, ['ANSWER:\n' + str(v) for v in batch['labels']])
                bit = ce[:, -2].mean(); formatting = .1 * torch.cat([ce[:, :-2], ce[:, -1:]], 1).mean()
                actor = batch['actor']; alogits, ace, at = forward_targets(tokenizer, model, [e['prompt'] for e in actor], [e['answer'] for e in actor])
                abit, aformat = set_loss(alogits, ace, [letters[e['good_indices']] for e in actor]); loss = bit + formatting + abit + aformat
                pt = 0; pbit = pformat = None
                if kind == 'proof':
                    proof_texts = [render(data[key], pool, name, 'proof')[0] for key, name in zip(batch['ids'], batch['names'])]
                    plogits, pce, pt = forward_targets(tokenizer, model, proof_texts, ['PROOF: ' + words[0] for words in valid])
                    allowed = [[tokenizer.encode(' ' + word, add_special_tokens=False)[0] for word in words] for words in valid]
                    pbit, pformat = set_loss(plogits, pce, allowed); loss = loss + pbit + pformat
                optimizer.zero_grad(); loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)); optimizer.step(); timer.cancel()
                assert {int(s['step']) for s in optimizer.state.values() if 'step' in s} == {step + 1}
                append(target / 'training.jsonl', {k: batch[k] for k in ['step', 'stratum', 'ids', 'names', 'labels', 'replay_source_step']} |
                       {'optimizer_step': step + 1, 'actor_ids': [e['id'] for e in actor], 'valid_witnesses': valid,
                        'loss': float(loss.detach()), 'bit_loss': float(bit.detach()), 'format_loss': float(formatting.detach()),
                        'actor_bit_loss': float(abit.detach()), 'actor_format_loss': float(aformat.detach()),
                        'proof_bit_loss': None if pbit is None else float(pbit.detach()), 'proof_format_loss': None if pformat is None else float(pformat.detach()),
                        'query_tokens': qt, 'actor_tokens': at, 'proof_tokens': pt, 'seconds': time.perf_counter() - start, 'gradient_norm': norm})
                if (step + 1) % 24 == 0: print(json.dumps({'arm': arm, 'step': step + 1}), flush=True)
                del logits, ce, alogits, ace, bit, formatting, abit, aformat, loss
                if kind == 'proof': del plogits, pce, pbit, pformat
            model.save_pretrained(target / 'adapter'); del optimizer
            (target / 'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes': {p.name: sha(p) for p in (target / 'adapter').iterdir() if p.is_file()},
                'steps': 192, 'initial_adapter_hashes': cp['seal']['adapter_hashes']}, indent=2))
        else:
            source = ROOT / cp['path']; shutil.copytree(source / 'adapter', target / 'adapter'); shutil.copy2(source / 'CHECKPOINT_SEALED.json', target / 'CHECKPOINT_SEALED.json')
        model.eval(); before = digest(model)
        for mode, cases in [('binary', obs), ('proof', obs), ('oracle', oracle)]:
            if mode == 'oracle' and kind == 'start':
                source = ROOT / cp['path'] / 'oracle.jsonl'; shutil.copy2(source, target / 'oracle.jsonl')
                (target / 'REUSED_ORACLE.json').write_text(json.dumps({'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source), 'new_calls': 0})); continue
            key = 'proof_prompt' if mode == 'proof' else 'prompt'
            for start in range(0, len(cases), 4):
                batch = cases[start:start + 4]; texts = [c[key] for c in batch]
                lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': t}], tokenize=True, add_generation_prompt=True)) for t in texts]
                raw, seconds = generate(tokenizer, model, texts)
                for case, text, length in zip(batch, raw, lengths): append(target / (mode + '.jsonl'), {'id': case['id'], 'raw': text,
                    'prompt_tokens': length, 'padded_width': max(lengths), 'seconds_batch': seconds})
            print(json.dumps({'arm': arm, 'mode': mode, 'responses': len(cases)}), flush=True)
        after = digest(model); assert before == after
        (target / 'EVAL_IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'errors': 0}))
        del model; gc.collect()
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()}, 'whole_goal_complete': False}, indent=2))


if __name__ == '__main__': main()
