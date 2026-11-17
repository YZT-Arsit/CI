"""Restored-parameter one-step query/action loss intervention, CPU only."""
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(R.parent / '20261117_frozen_crossover'), str(R.parent / '20261117_teacher_binding'),
                str(R.parent / '20261117_sampler_repair')]
from common import verify, load, torch, palette
from binding_loss import loss_for_views
from sampling import buckets


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((R / 'PLAN.json').read_text())
    assert not (R / 'RUN_PLAN.json').exists()
    assert all(sha(R / name) == digest for name, digest in plan['sources'].items())
    assert all(sha(ROOT / name) == digest for name, digest in plan['dependencies'].items())
    cross = verify()
    assert sha(R.parent / '20261117_frozen_crossover/PLAN.json') == plan['cross_plan_sha256']
    assert sha(R.parent / '20261117_teacher_binding/PLAN.json') == plan['old_plan_sha256']
    assert cross['checkpoints']['teacher_sft'] == plan['checkpoint']
    tokenizer, model = load('teacher_sft', cross)
    model.eval()  # LoRA dropout is zero; all branches use this same mode.
    params = [p for p in model.parameters() if p.requires_grad]
    originals = [p.detach().clone() for p in params]
    pool = palette(tokenizer)['train']
    data = buckets()
    timers = []

    def digest():
        h = hashlib.sha256()
        for p in params:
            h.update(p.detach().contiguous().numpy().tobytes())
        return h.hexdigest()

    initial = digest()

    def restore():
        with torch.no_grad():
            for p, value in zip(params, originals):
                p.copy_(value)
                p.grad = None
        assert digest() == initial

    def timed(label, fn):
        def timeout():
            (R / 'TIMEOUT.json').write_text(json.dumps({'operation': label, 'limit': 15}))
            os._exit(124)
        timer = threading.Timer(plan['max_operation_seconds'], timeout)
        timer.daemon = True
        timer.start()
        start = time.perf_counter()
        try:
            value = fn()
        finally:
            timer.cancel()
        timers.append({'operation': label, 'seconds': time.perf_counter() - start})
        return value

    def parts(batch, step):
        pair = data[tuple(batch['key'])][batch['index']]
        _, losses, _, _, _ = loss_for_views(tokenizer, model, pair, pool, batch['views'], step, 0., teacher=True)
        return losses['query_loss'], losses['actor_loss']

    def evaluate(batch, step):
        with torch.no_grad():
            values = [float(x) for x in parts(batch, step)]
        assert all(torch.isfinite(torch.tensor(values)))
        return dict(zip(['query', 'actor'], values))

    (R / 'RUN_PLAN.json').write_text(json.dumps({'plan_sha256': sha(R / 'PLAN.json'),
        'initial_parameter_sha256': initial, 'persistent_checkpoint_written': False}, indent=2))
    out = R / 'records.jsonl'
    for index, case in enumerate(plan['cases']):
        restore()

        def gradients():
            q, a = parts(case['anchor'], case['anchor_step'])
            gq = torch.autograd.grad(q, params, retain_graph=True)
            ga = torch.autograd.grad(a, params)
            assert all(torch.isfinite(g).all() for g in (*gq, *ga))
            return q.item(), a.item(), [g.detach() for g in gq], [g.detach() for g in ga]

        q, a, gq, ga = timed(f'{index}/gradients', gradients)
        dot = sum(float((x.double()*y.double()).sum()) for x, y in zip(gq, ga))
        nq = sum(float(x.double().square().sum()) for x in gq)
        na = sum(float(x.double().square().sum()) for x in ga)
        before = {'query': q, 'actor': a}
        companion = timed(f'{index}/companion_before', lambda: evaluate(case['companion'], case['companion_step']))
        branches = []
        for arm in plan['arms']:
            restore()
            if arm != 'none':
                optimizer = torch.optim.AdamW(params, lr=plan['lr'], weight_decay=0.)
                assert not optimizer.state
                for p, x, y in zip(params, gq, ga):
                    p.grad = (x if arm == 'query' else y if arm == 'actor' else x+y).clone()
                norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
                optimizer.step()
                del optimizer
            else:
                norm = 0.
            after = timed(f'{index}/{arm}/anchor', lambda: evaluate(case['anchor'], case['anchor_step']))
            other = timed(f'{index}/{arm}/companion', lambda: evaluate(case['companion'], case['companion_step']))
            if arm == 'none':
                assert max(abs(after[k]-before[k]) for k in before) < 1e-6
                assert max(abs(other[k]-companion[k]) for k in before) < 1e-6
            branches.append({'arm': arm, 'gradient_norm_before_clip': norm,
                             'parameter_sha256': digest(), 'anchor_after': after, 'companion_after': other})
        row = {'index': index, 'key': case['key'], 'anchor_before': before,
               'companion_before': companion, 'query_norm2': nq, 'actor_norm2': na, 'dot': dot,
               'cosine': dot/(nq*na)**.5 if nq*na else None, 'branches': branches}
        with out.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps({'case': index+1, 'total': len(plan['cases']), 'cosine': row['cosine']}), flush=True)
    restore()
    (R / 'TIMINGS.json').write_text(json.dumps(timers, indent=2))
    (R / 'SEALED.json').write_text(json.dumps({'complete': True, 'restored_parameter_sha256': digest(),
        'files': {p: sha(R / p) for p in ['records.jsonl', 'TIMINGS.json']},
        'whole_goal_complete': False}, indent=2))


if __name__ == '__main__':
    main()
