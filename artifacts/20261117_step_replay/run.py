"""Three matched local-plus-rehearsal updates; seven frozen final evaluations."""
import gc
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from replay_data import R, ROOT, PARENT, NAMES, read, rows, sha, verify, prompt
sys.path[:0] = [str(R.parent / '20261117_frozen_crossover'), str(R.parent / '20261117_teacher_binding')]
from common import load, generate, verify as verify_model, torch
from binding_loss import forward_targets, answer_ids


def append(path, row):
    with path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row) + '\n')


def digest(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad: h.update(name.encode()); h.update(parameter.detach().numpy().tobytes())
    return h.hexdigest()


def main():
    plan = verify(); verify_model(); folder = R / 'study'; assert not folder.exists(); folder.mkdir()
    data = read(R / 'training_worlds.json'); pool = read(NAMES / 'palettes.json')['train']
    obs = rows(R / 'observations.jsonl'); oracle = rows(R.parent / '20261117_contrast_repair/fresh_observations.jsonl')
    for arm in plan['arms']:
        target = folder / arm; target.mkdir(); trained = arm.startswith('replay')
        cp = plan['checkpoints']['initial' if trained else arm]
        tokenizer, model = load('start', {'checkpoints': {'start': cp}})
        (target / 'INITIAL_STATE.json').write_text(json.dumps({'trainable_sha256': digest(model), 'checkpoint': cp['path']}))
        if trained:
            seed = int(arm.removeprefix('replay')); local = rows(R / f'local_{seed}.jsonl'); replay = rows(R / f'replay_{seed}.jsonl')
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['lr'], weight_decay=0.)
            letters = answer_ids(tokenizer)
            for step, (batch, rehearsal) in enumerate(zip(local, replay)):
                def timeout():
                    (target / 'TIMEOUT.json').write_text(json.dumps({'step': step + 1, 'seconds': 15})); os._exit(124)
                timer = threading.Timer(15, timeout); timer.daemon = True; timer.start(); start = time.perf_counter(); model.train()
                texts = [prompt(data[key], pool, name) for key, name in zip(batch['ids'], batch['names'])]
                answers = ['ANSWER:\n' + str(bit) for bit in batch['labels']]
                logits, ce, local_tokens = forward_targets(tokenizer, model, texts, answers)
                bit = ce[:, -2].mean(); formatting = .1 * torch.cat([ce[:, :-2], ce[:, -1:]], 1).mean()
                examples = rehearsal['examples']
                actor_logits, ace, actor_tokens = forward_targets(tokenizer, model, [e['prompt'] for e in examples], [e['answer'] for e in examples])
                action_bit = torch.stack([torch.logsumexp(actor_logits[i, -2], 0) - torch.logsumexp(actor_logits[i, -2, letters[e['good_indices']]], 0) for i, e in enumerate(examples)]).mean()
                action_format = .1 * torch.cat([ace[:, :-2], ace[:, -1:]], 1).mean()
                actor_loss = action_bit + action_format; loss = bit + formatting + plan['actor_weight'] * actor_loss
                optimizer.zero_grad(); loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                optimizer.step(); timer.cancel(); assert {int(s['step']) for s in optimizer.state.values() if 'step' in s} == {step + 1}
                append(target / 'training.jsonl', {**batch, 'replay_ids': [e['id'] for e in examples], 'good_indices': [e['good_indices'] for e in examples],
                       'optimizer_step': step + 1, 'loss': float(loss.detach()), 'bit_loss': float(bit.detach()), 'format_loss': float(formatting.detach()),
                       'actor_loss': float(actor_loss.detach()), 'actor_bit_loss': float(action_bit.detach()), 'actor_format_loss': float(action_format.detach()),
                       'local_tokens': local_tokens, 'actor_tokens': actor_tokens, 'seconds': time.perf_counter() - start, 'gradient_norm': norm})
                if (step + 1) % 24 == 0: print(json.dumps({'arm': arm, 'step': step + 1}), flush=True)
                del logits, ce, bit, formatting, actor_logits, ace, action_bit, action_format, actor_loss, loss
            model.save_pretrained(target / 'adapter'); del optimizer
            (target / 'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes': {p.name: sha(p) for p in (target / 'adapter').iterdir() if p.is_file()},
                'steps': 144, 'initial_adapter_hashes': cp['seal']['adapter_hashes']}, indent=2))
        else:
            source = ROOT / cp['path']; shutil.copytree(source / 'adapter', target / 'adapter'); shutil.copy2(source / 'CHECKPOINT_SEALED.json', target / 'CHECKPOINT_SEALED.json')
        model.eval(); before = digest(model)
        for mode, cases in [('operator', obs), ('oracle', oracle)]:
            if mode == 'oracle' and not trained:
                source = ROOT / cp['path'] / 'oracle.jsonl'; shutil.copy2(source, target / 'oracle.jsonl')
                (target / 'REUSED_ORACLE.json').write_text(json.dumps({'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source), 'new_calls': 0})); continue
            for start in range(0, len(cases), 4):
                batch = cases[start:start + 4]
                lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': c['prompt']}], tokenize=True, add_generation_prompt=True)) for c in batch]
                raw, seconds = generate(tokenizer, model, [c['prompt'] for c in batch])
                for case, text, length in zip(batch, raw, lengths):
                    append(target / (mode + '.jsonl'), {'id': case['id'], 'raw': text, 'prompt_tokens': length, 'padded_width': max(lengths), 'seconds_batch': seconds})
            print(json.dumps({'arm': arm, 'mode': mode, 'responses': len(cases)}), flush=True)
        after = digest(model); assert before == after
        (target / 'EVAL_IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'errors': 0}))
        del model; gc.collect()
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()}, 'whole_goal_complete': False}, indent=2))


if __name__ == '__main__': main()
