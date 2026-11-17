"""Local four-component gradients; no optimizer and no causal efficacy claim."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261117_binding_factors'))
from repair import *
HERE = Path(__file__).resolve().parent


def main():
    assert not (HERE / 'GRADIENT_PLAN.json').exists()
    prior = HERE.parent / '20261117_binding_factors/repair'
    training_plan = json.loads((prior / 'PLAN.json').read_text())
    assert hashlib.sha256((prior.parent / 'repair.py').read_bytes()).hexdigest() == training_plan['source_sha256']
    assert all(hashlib.sha256((prior.parent / p).read_bytes()).hexdigest() == h
               for p, h in training_plan['dependency_hashes'].items())
    frozen_checkpoints = json.loads((HERE / 'RUN_PLAN.json').read_text())['checkpoints']
    selected = []
    for role in ['Planner', 'Source', 'Worker']:
        selected += [(i, b) for i, b in enumerate(training_plan['batches']) if b['key'][0] == role][:2]
    checkpoints = {'initial': HERE.parent / '20261117_action_binding/coverage_repair/class_balanced',
                   'two_view_sft': prior / 'two_view_sft', 'conditional_js': prior / 'conditional_js'}
    plan = {'batches': [{'original_step_index': i, **b} for i, b in selected],
            'selection': 'first two original training batches per role, independent of diagnostic outcomes',
            'components': ['query_loss', 'actor_loss', 'query_js', 'actor_js'],
            'updates': 0, 'trial_watchdog_seconds': 15,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_plan_sha256': hashlib.sha256((prior / 'PLAN.json').read_bytes()).hexdigest(),
            'dependency_hashes': training_plan['dependency_hashes'],
            'scope': 'Local representative TRAIN gradients only, not a causal explanation or whole-distribution claim'}
    (HERE / 'GRADIENT_PLAN.json').write_text(json.dumps(plan, indent=2))
    buckets = full_buckets()
    records = []
    for arm, folder in checkpoints.items():
        seal = json.loads((folder / 'CHECKPOINT_SEALED.json').read_text())
        assert seal['adapter_hashes'] == (training_plan['initial_adapter_hashes'] if arm == 'initial'
                                          else frozen_checkpoints[arm]['adapter_hashes'])
        assert all(hashlib.sha256((folder / 'adapter' / p).read_bytes()).hexdigest() == h
                   for p, h in seal['adapter_hashes'].items())
        tokenizer, model = setup(17)
        set_peft_model_state_dict(model, load_file(str(folder / 'adapter/adapter_model.safetensors')))
        model.train()
        pool = palette(tokenizer)['train']
        params = [p for p in model.parameters() if p.requires_grad]
        for step, batch in selected:
            def timeout():
                (HERE / 'GRADIENT_TIMEOUT.json').write_text(json.dumps({'arm': arm, 'step': step, 'seconds': 15}))
                os._exit(124)
            timer = threading.Timer(15, timeout)
            timer.daemon = True
            timer.start()
            start = time.perf_counter()
            pair = buckets[tuple(batch['key'])][batch['index']]
            loss, parts, tokens, estimates, eligible = loss_for_views(tokenizer, model, pair, pool, batch['views'], step, 1.)
            gradients = {}
            for i, name in enumerate(plan['components']):
                pieces = torch.autograd.grad(parts[name], params, retain_graph=i < 3, allow_unused=True)
                assert all(g is None or torch.isfinite(g).all() for g in pieces)
                gradients[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                                              for p, g in zip(params, pieces)])
            gradients['supervised'] = gradients['query_loss'] + gradients['actor_loss']
            gradients['consistency'] = gradients['query_js'] + gradients['actor_js']
            norms = {name: float(g.norm()) for name, g in gradients.items()}
            dots = {}
            cosines = {}
            for left, right in [('query_loss', 'actor_loss'), ('query_loss', 'query_js'),
                                 ('actor_loss', 'actor_js'), ('supervised', 'consistency')]:
                key = left + '__' + right
                dots[key] = float(torch.dot(gradients[left], gradients[right]))
                denominator = norms[left] * norms[right]
                cosines[key] = dots[key] / denominator if denominator else None
            timer.cancel()
            record = {'arm': arm, 'original_step_index': step, 'role': batch['key'][0],
                       'batch_key': batch['key'], 'losses': {n: float(p.detach()) for n, p in parts.items()},
                       'norms': norms, 'dots': dots, 'cosines': cosines, 'eligible_states': eligible,
                       'estimates': estimates, 'seconds': time.perf_counter() - start,
                       'adapter_hashes': seal['adapter_hashes'], 'all_gradients_finite': True}
            records.append(record)
            with (HERE / 'gradients.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps({k: record[k] for k in ['arm', 'role', 'original_step_index', 'seconds', 'cosines']}), flush=True)
            del loss, parts, gradients, pieces
        del model
        gc.collect()
    (HERE / 'GRADIENT_RESULTS.json').write_text(json.dumps({
        'trials': len(records), 'updates': 0, 'max_trial_seconds': max(r['seconds'] for r in records),
        'all_gradients_finite': True, 'raw_sha256': hashlib.sha256((HERE / 'gradients.jsonl').read_bytes()).hexdigest(),
        'scope': plan['scope']}, indent=2))


if __name__ == '__main__':
    main()
