import importlib.util
import json
import math
from rehearsal_data import R, ROOT, PARENT, read, rows, sha, verify, training, worlds

spec = importlib.util.spec_from_file_location('join_scores_for_rehearsal', PARENT / 'audit_join.py')
scoring = importlib.util.module_from_spec(spec); spec.loader.exec_module(scoring)


def main():
    plan = verify(); seal = read(R / 'SEALED.json'); token = read(R / 'TOKEN_PREFLIGHT.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json') and all(sha(R / n) == h for n, h in seal['files'].items())
    assert token['plan_sha256'] == sha(R / 'PLAN.json') and token['script_sha256'] == sha(R / 'token_preflight.py')
    assert worlds()[0] == read(R / 'dev_worlds.json')
    runtime = read(R / 'study/PLAN.json')
    assert runtime == {**plan, 'arms': list(plan['arms']), 'evidence_arms': list(plan['arms']),
        'checkpoint_bindings': runtime['checkpoint_bindings'], 'source_sha256': sha(R / 'run_rehearsal.py')}
    assert (R / 'study/dev_worlds.json').read_bytes() == (R / 'dev_worlds.json').read_bytes()
    schedules = {s: training(s) for s in [17, 29, 41]}
    for s, schedule in schedules.items(): assert schedule == rows(R / f'training_{s}.jsonl')
    summaries = []; train_times = []; times = []; new = reused = 0; tokens = []
    for arm, cfg in plan['arms'].items():
        seed, kind = cfg['seed'], cfg['kind']; folder = R / 'study' / arm; cp = plan['checkpoints'][str(seed)]
        initial = read(folder / 'INITIAL_STATE.json'); ident = read(folder / 'EVAL_IDENTITY.json')
        assert initial == {'digest': read(ROOT / cp['path'] / 'EVAL_IDENTITY.json')['before'], 'checkpoint': cp}
        assert ident['before'] == ident['after'] and ident['errors'] == 0
        binding = cp if kind == 'initial' else {'path': folder.relative_to(ROOT).as_posix(), 'seal': read(folder / 'CHECKPOINT_SEALED.json')}
        assert ident['checkpoint'] == runtime['checkpoint_bindings'][arm] == binding
        assert (ident['before'] == initial['digest']) == (kind == 'initial')
        if kind == 'initial':
            assert not (folder / 'adapter').exists(); reuse = read(folder / 'REUSED.json')
            for p in ['standard', 'factors', 'oracle']:
                source = ROOT / cp['path'] / (p + '.jsonl')
                assert source.read_bytes() == (folder / (p + '.jsonl')).read_bytes()
                assert reuse[p] == {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source), 'new_calls': 0}
            reused += 912
        else:
            saved = binding['seal']; assert saved['steps'] == 54 and saved['initial_adapter_hashes'] == cp['seal']['adapter_hashes']
            assert all(sha(folder / 'adapter' / n) == h for n, h in saved['adapter_hashes'].items())
            updates = rows(folder / 'training.jsonl'); assert len(updates) == 54 and not (folder / 'TIMEOUT.json').exists()
            for i, (record, batch) in enumerate(zip(updates, schedules[seed])):
                assert record['step'] == i and record['optimizer_step'] == i + 1 and record['global_cell'] == batch['global_cell']
                expected = token['training_tokens'][str(seed)][i]
                assert record['tokens'] == {k: expected[k] for k in ['common', kind, 'actor']}
                assert len(record['losses']) == 6 and all(math.isfinite(v) for v in [*record['losses'], record['loss'], record['gradient_norm']])
                assert abs(record['loss'] - sum(record['losses'])) < 1e-5 and 0 < record['seconds'] < 15
                train_times.append(record['seconds'])
            tokens.append({'arm': arm, **{k: sum(r['tokens'][k] for r in updates) for k in ['common', kind, 'actor']}})
        scores = {}
        for p, score in [('standard', scoring.standard_score), ('factors', scoring.factor_score)]:
            obs = rows(PARENT / f'{p}_observations.jsonl'); labels = rows(PARENT / f'{p}_labels.jsonl'); raw = rows(folder / f'{p}.jsonl')
            scores[p], elapsed = score(obs, labels, raw, 'binary')
            assert [r['prompt_tokens'] for r in raw] == token['eval_lengths'][p]
            if kind != 'initial': times.extend(elapsed); new += len(raw)
        actor, elapsed = scoring.oracle_score(folder)
        if kind != 'initial': times.extend(elapsed); new += 144
        summaries.append({'arm': arm, 'seed': seed, 'kind': kind, **scores, 'oracle': actor})
    assert new == 5472 and reused == 2736 and len(train_times) == 324
    result = {'summaries': summaries, 'new_responses': new, 'reused_responses': reused, 'new_updates': 324, 'new_checkpoints': 6,
              'tokens': tokens, 'training_seconds': sum(train_times), 'max_update_seconds': max(train_times),
              'generation_seconds': sum(times), 'max_generation_seconds': max(times), 'errors': 0, 'whole_goal_complete': False}
    (R / 'LOCAL_RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'summaries'}))


if __name__ == '__main__': main()
