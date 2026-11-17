"""Check the exact loss change and real-model equivalence before training."""
import hashlib
import json
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import value_loss as new
from two_view_loss import loss_for_views as original
from coverage_repair import setup, palette, load_file, set_peft_model_state_dict
sys.path.insert(0, str(R.parent / '20261117_sampler_repair'))
from sampling import buckets


def main():
    assert not (R / 'PREFLIGHT.json').exists()
    torch = new.torch
    values = [(1, 1, 0), (1, 1, 0), (0, 0, 0)]
    p = torch.tensor([.8, .1, .08, .02], dtype=torch.float64).log()
    q = torch.tensor([.1, .8, .08, .02], dtype=torch.float64).log()
    assert new.js(p, q) > .3
    assert abs(new.js(new.value_log_probs(p, values), new.value_log_probs(q, values)).item()) < 1e-12
    assert torch.equal(new.value_log_probs(p, values)[-1], p[-1])
    unique = [(1, 1, 0), (1, 0, 0), (0, 0, 0)]
    assert abs((new.js(p, q) - new.js(new.value_log_probs(p, unique), new.value_log_probs(q, unique))).item()) < 1e-12
    a = torch.tensor([1., 2., 3., -1.], requires_grad=True)
    b = torch.tensor([3., 1., 0., -1.], requires_grad=True)
    loss = new.js(new.value_log_probs(a.log_softmax(0), values), new.value_log_probs(b.log_softmax(0), values))
    loss.backward()
    assert a.grad.norm() > 0 and b.grad.norm() > 0
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    assert abs(new.value_log_probs(p, values).exp().sum().item() - 1) < 1e-12
    tokenizer, model = setup(17)
    initial = R.parent / '20261117_action_binding/coverage_repair/class_balanced'
    seal = json.loads((initial / 'CHECKPOINT_SEALED.json').read_text())
    assert all(hashlib.sha256((initial / 'adapter' / name).read_bytes()).hexdigest() == digest for name, digest in seal['adapter_hashes'].items())
    set_peft_model_state_dict(model, load_file(str(initial / 'adapter/adapter_model.safetensors')))
    model.eval()
    pool = palette(tokenizer)['train']
    data = buckets()
    plan = json.loads((R.parent / '20261117_sampler_repair/PLAN.json').read_text())
    trials = []
    for role in ['Planner', 'Source', 'Worker']:
        step, batch = next((i, item) for i, item in enumerate(plan['batches']) if item['key'][0] == role)
        pair = data[tuple(batch['key'])][batch['index']]
        start = time.perf_counter()
        with torch.no_grad():
            old = original(tokenizer, model, pair, pool, batch['views'], step, 1.)
            full = new.loss_for_views(tokenizer, model, pair, pool, batch['views'], step, 1., 'full')
            assert torch.equal(old[0], full[0])
            assert all(torch.equal(old[1][k], full[1][k]) for k in old[1])
            assert old[2:] == full[2:]
        equal_seconds = time.perf_counter() - start
        assert equal_seconds < 15
        for mode in ['none', 'value']:
            start = time.perf_counter()
            model.zero_grad(set_to_none=True)
            out = new.loss_for_views(tokenizer, model, pair, pool, batch['views'], step, 1., mode)
            assert torch.isfinite(out[0])
            out[0].backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            elapsed = time.perf_counter() - start
            assert elapsed < 9.5
            trials.append({'role': role, 'mode': mode, 'seconds': elapsed, 'full_loss_exact_match_seconds': equal_seconds,
                           'parts': {k: v.item() for k, v in out[1].items()}, 'gradient_tensors': len(gradients)})
    result = {'mathematical_checks': 'PASS', 'all_roles_full_loss_exact_match': True,
              'finite_gradients_both_variants': True, 'optimizer_updates': 0, 'trials': trials,
              'initial_adapter_hashes': seal['adapter_hashes'],
              'loss_sha256': hashlib.sha256((R / 'value_loss.py').read_bytes()).hexdigest()}
    (R / 'PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
