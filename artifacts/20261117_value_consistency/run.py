"""Loss-only matched ablation; reuse two baseline checkpoints."""
import shutil
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / '20261117_binding_factors'))
from repair import *
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(OUT.parent / '20261117_sampler_repair'))
from sampling import buckets as pure_buckets, DATA
sys.path.insert(0, str(OUT))
from value_loss import loss_for_views


def main():
    specification = json.loads((OUT / 'PLAN.json').read_text())
    folder = OUT / 'repair'
    assert not (folder / 'PLAN.json').exists()
    folder.mkdir(exist_ok=True)
    old = OUT.parent / '20261117_sampler_repair/repair'
    assert hashlib.sha256((old.parent / 'PLAN.json').read_bytes()).hexdigest() == specification['prior_plan_sha256']
    assert hashlib.sha256((OUT.parent / '20261117_binding_factors/two_view_loss.py').read_bytes()).hexdigest() == specification['old_loss_sha256']
    preflight = json.loads((OUT / 'PREFLIGHT.json').read_text())
    assert preflight['all_roles_full_loss_exact_match'] and preflight['finite_gradients_both_variants']
    assert preflight['loss_sha256'] == hashlib.sha256((OUT / 'value_loss.py').read_bytes()).hexdigest()
    old_plan = json.loads((old / 'PLAN.json').read_text())
    for name, digest in specification['source_hashes'].items():
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest
    for name, digest in specification['training_files'].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest
    for name, digest in old_plan['unchanged_loss_dependencies'].items():
        assert hashlib.sha256((OUT.parent / '20261117_binding_factors' / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((OUT / 'dev_worlds.json').read_bytes()).hexdigest() == specification['dev_sha256']
    data = full_buckets()
    assert data == pure_buckets()
    plan = {**specification, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'unchanged_loss_dependencies': old_plan['unchanged_loss_dependencies'],
            'baseline_seals': {arm: json.loads((old / origin / 'CHECKPOINT_SEALED.json').read_text())
                               for arm, origin in specification['baseline_checkpoints'].items()}}
    (folder / 'PLAN.json').write_text(json.dumps(plan, indent=2))
    shutil.copy2(OUT / 'dev_worlds.json', folder / 'dev_worlds.json')
    worlds = json.loads((folder / 'dev_worlds.json').read_text())
    summaries = []
    initial = OUT.parent / '20261117_action_binding/coverage_repair/class_balanced/adapter'
    for arm in ['query_js', 'value_js', 'sampler_sft', 'full_js']:
        target = folder / arm
        target.mkdir()
        tokenizer, model = setup(17)
        baseline = arm in plan['baseline_checkpoints']
        source = old / plan['baseline_checkpoints'][arm] / 'adapter' if baseline else initial
        expected = plan['baseline_seals'][arm]['adapter_hashes'] if baseline else plan['initial_adapter_hashes']
        assert all(hashlib.sha256((source / p).read_bytes()).hexdigest() == h for p, h in expected.items())
        set_peft_model_state_dict(model, load_file(str(source / 'adapter_model.safetensors')))
        pool = palette(tokenizer)
        if baseline:
            shutil.copytree(source, target / 'adapter')
            shutil.copy2(source.parent / 'CHECKPOINT_SEALED.json', target / 'CHECKPOINT_SEALED.json')
            (target / 'ORIGIN.json').write_text(json.dumps({'reused_checkpoint': source.relative_to(OUT.parent.parent).as_posix(), 'new_optimizer_updates': 0}))
        else:
            model.train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=plan['lr'], weight_decay=0.)
            tokens_total = 0
            for step, batch in enumerate(plan['batches']):
                def timeout():
                    (target / 'TIMEOUT.json').write_text(json.dumps({'step': step, 'seconds': 9.5}))
                    os._exit(124)
                timer = threading.Timer(9.5, timeout)
                timer.daemon = True
                timer.start()
                start = time.perf_counter()
                pair = data[tuple(batch['key'])][batch['index']]
                loss, parts, tokens, estimates, eligible = loss_for_views(tokenizer, model, pair, pool['train'], batch['views'], step, 1., plan['actor_modes'][arm])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                timer.cancel()
                tokens_total += tokens
                record = {'step': step + 1, 'role': batch['key'][0],
                           **{k: float(v.detach()) for k, v in parts.items()}, 'input_tokens': tokens,
                           'actor_consistency_eligible_states': eligible, 'predicted_estimates': estimates,
                           'seconds': time.perf_counter() - start}
                with (target / 'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                if (step + 1) % 20 == 0:
                    print(json.dumps({'arm': arm, **{k: v for k, v in record.items() if k != 'predicted_estimates'}}), flush=True)
            model.save_pretrained(target / 'adapter')
            (target / 'CHECKPOINT_SEALED.json').write_text(json.dumps({
                'adapter_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (target / 'adapter').iterdir() if p.is_file()},
                'steps': 80, 'input_tokens': tokens_total}, indent=2))
            del optimizer
        model.eval()
        for condition, ws in worlds.items():
            records = evaluate(tokenizer, model, ws, pool['fresh'], True)
            path = target / (condition + '.jsonl')
            path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
            summary = {'arm': arm, 'condition': condition, 'episodes': len(records),
                        **{k: sum(r[k] for r in records) for k in ['safe_success', 'strict_success', 'invalid', 'invalid_queries']},
                        'valid_ci': sum(r['ci'] and not r['invalid'] for r in records),
                        'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        del model
        gc.collect()
    (folder / 'RESULTS.json').write_text(json.dumps({'summaries': summaries, 'whole_goal_complete': False}, indent=2))


if __name__ == '__main__':
    main()
