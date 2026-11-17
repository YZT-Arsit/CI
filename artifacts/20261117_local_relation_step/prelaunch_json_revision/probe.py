"""Frozen short CPU inference; no labels used outside the declared demonstrations."""
import gc
import hashlib
import json
import sys
from step_data import R, ROOT, read, rows, sha, verify
sys.path.insert(0, str(R.parent / '20261117_frozen_crossover'))
from common import load, generate, verify as verify_model


def digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad: h.update(name.encode()); h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def main():
    plan = verify(); verify_model(); folder = R / 'probe'; assert not folder.exists(); folder.mkdir()
    observations = rows(R / 'observations.jsonl')
    for arm in plan['arms']:
        target = folder / arm; target.mkdir(); tokenizer, model = load(arm, plan); model.eval(); before = digest(model)
        for mode in ['zero_shot', 'four_examples']:
            obs = [o for o in observations if o['mode'] == mode]
            for start in range(0, len(obs), 4):
                batch = obs[start:start + 4]
                lengths = [len(tokenizer.apply_chat_template([{'role': 'user', 'content': o['prompt']}], tokenize=True, add_generation_prompt=True)) for o in batch]
                raw, seconds = generate(tokenizer, model, [o['prompt'] for o in batch])
                with (target / (mode + '.jsonl')).open('a', encoding='utf-8') as stream:
                    for o, text, length in zip(batch, raw, lengths):
                        stream.write(json.dumps({'id': o['id'], 'raw': text, 'prompt_tokens': length, 'padded_width': max(lengths), 'seconds_batch': seconds}) + '\n')
            print(json.dumps({'arm': arm, 'mode': mode, 'responses': len(obs)}), flush=True)
        after = digest(model); assert before == after
        (target / 'IDENTITY.json').write_text(json.dumps({'before': before, 'after': after, 'errors': 0}))
        del model; gc.collect()
    (R / 'PROBE_SEALED.json').write_text(json.dumps({'complete': True, 'plan_sha256': sha(R / 'PLAN.json'),
        'files': {p.relative_to(R).as_posix(): sha(p) for p in folder.rglob('*') if p.is_file()}, 'whole_goal_complete': False}, indent=2))


if __name__ == '__main__': main()
