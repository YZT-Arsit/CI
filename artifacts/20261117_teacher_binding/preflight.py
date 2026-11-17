"""Check own-input equivalence and teacher-map integrity before parameter updates."""
import hashlib
import json
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import binding_loss as new
sys.path.insert(0, str(R.parent / '20261117_value_consistency'))
from value_loss import loss_for_views as old
from coverage_repair import setup, palette, load_file, set_peft_model_state_dict
sys.path.insert(0, str(R.parent / '20261117_sampler_repair'))
from sampling import buckets


def main():
    assert not (R / 'PREFLIGHT.json').exists()
    tokenizer, model = setup(17)
    initial = R.parent / '20261117_action_binding/coverage_repair/class_balanced'
    seal = json.loads((initial / 'CHECKPOINT_SEALED.json').read_text())
    assert all(hashlib.sha256((initial / 'adapter' / name).read_bytes()).hexdigest() == digest for name, digest in seal['adapter_hashes'].items())
    set_peft_model_state_dict(model, load_file(str(initial / 'adapter/adapter_model.safetensors')))
    model.eval()
    pool = palette(tokenizer)['train']
    batches = json.loads((R.parent / '20261117_sampler_repair/PLAN.json').read_text())['batches']
    data, trials = buckets(), []
    for role in ['Planner', 'Source', 'Worker']:
        step, batch = next((i, b) for i, b in enumerate(batches) if b['key'][0] == role)
        pair = data[tuple(batch['key'])][batch['index']]
        for weight in (0., 1.):
            start = time.perf_counter()
            with new.torch.no_grad():
                before = old(tokenizer, model, pair, pool, batch['views'], step, weight, 'none')
                own = new.loss_for_views(tokenizer, model, pair, pool, batch['views'], step, weight, False)
                assert new.torch.equal(before[0], own[0])
                assert all(new.torch.equal(before[1][k], own[1][k]) for k in before[1])
                assert before[2:4] == own[2:4] and own[3] == own[4]
            equivalence_seconds = time.perf_counter() - start
            assert equivalence_seconds < 15
            model.zero_grad(set_to_none=True)
            start = time.perf_counter()
            teacher = new.loss_for_views(tokenizer, model, pair, pool, batch['views'], step, weight, True)
            expected = [[{c['action']: str(c['safe_completion']) for c in row['candidates']} for row in pair] for _ in batch['views']]
            assert teacher[4] == expected and teacher[3] == own[3]
            assert all(new.torch.equal(teacher[1][k], own[1][k]) for k in ['query_loss', 'query_js'])
            assert new.torch.isfinite(teacher[0])
            teacher[0].backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            assert len(grads) == 120 and all(g is not None and new.torch.isfinite(g).all() for g in grads)
            seconds = time.perf_counter() - start
            assert seconds < 9.5
            trials.append({'role': role, 'query_weight': weight, 'seconds': seconds,
                           'equivalence_seconds': equivalence_seconds,
                           'actor_loss_own': own[1]['actor_loss'].item(), 'actor_loss_teacher': teacher[1]['actor_loss'].item(),
                           'changed_maps': sum(a != b for av, bv in zip(teacher[3], teacher[4]) for a, b in zip(av, bv))})
    result = {'own_mode_exact_match': True, 'teacher_gradients_finite': True, 'query_computation_unchanged': True,
              'optimizer_updates': 0, 'trials': trials, 'initial_adapter_hashes': seal['adapter_hashes'],
              'loss_sha256': hashlib.sha256((R / 'binding_loss.py').read_bytes()).hexdigest()}
    (R / 'PREFLIGHT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
