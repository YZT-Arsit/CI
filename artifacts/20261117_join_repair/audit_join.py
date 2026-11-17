import importlib.util
import json
import math
from join_data import R, ROOT, PARENT, FACTORS, read, rows, sha, verify, training, fresh_standard, fresh_factors


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


standard_score = module('standard_score_for_join', PARENT / 'scoring.py').score
factor_score = module('factor_score_for_join', FACTORS / 'scoring_factors.py').score
oracle_score = module('oracle_audit_for_join', R.parent / '20261117_step_learning/audit.py').oracle_score


def main():
    plan = verify(); seal = read(R / 'SEALED.json'); token = read(R / 'TOKEN_PREFLIGHT.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json') and all(sha(R / n) == h for n, h in seal['files'].items())
    assert token['plan_sha256'] == sha(R / 'PLAN.json') and token['script_sha256'] == sha(R / 'token_preflight.py')
    panels = {'standard': fresh_standard(), 'factors': fresh_factors()}
    for p, (obs, labels) in panels.items(): assert obs == rows(R / f'{p}_observations.jsonl') and labels == rows(R / f'{p}_labels.jsonl')
    summaries = []; times = []; train_times = []; totals = []; initials = {}; count = 0
    for arm, cfg in plan['arms'].items():
        seed, kind = cfg['seed'], cfg['kind']; folder = R / 'study' / arm; initial = read(folder / 'INITIAL_STATE.json'); cp = plan['checkpoints'][str(seed)]
        assert initial['checkpoint'] == cp and initial['digest'] == read(ROOT / cp['path'] / 'EVAL_IDENTITY.json')['before']
        identity = read(folder / 'EVAL_IDENTITY.json'); assert identity['before'] == identity['after'] and identity['errors'] == 0
        if kind == 'initial':
            initials[seed] = initial['digest']; assert identity['before'] == initial['digest']; source = ROOT / cp['path'] / 'oracle.jsonl'
            assert source.read_bytes() == (folder / 'oracle.jsonl').read_bytes()
            assert read(folder / 'REUSED_ORACLE.json') == {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source), 'new_calls': 0}
            assert not (folder / 'adapter').exists()
        else:
            assert initial['digest'] == initials[seed] and identity['before'] != initial['digest']
            saved = read(folder / 'CHECKPOINT_SEALED.json'); assert saved['steps'] == 96 and saved['initial_adapter_hashes'] == cp['seal']['adapter_hashes']
            assert all(sha(folder / 'adapter' / n) == h for n, h in saved['adapter_hashes'].items())
            data, schedule = training(seed); records = rows(folder / 'training.jsonl')
            assert data == read(R / f'worlds_{seed}.json') and schedule == rows(R / f'schedule_{seed}.jsonl')
            assert len(records) == 96 and not (folder / 'TIMEOUT.json').exists()
            for i, (record, batch) in enumerate(zip(records, schedule)):
                assert all(record[k] == batch[k] for k in ['step', 'stratum', 'ids', 'labels', 'names'])
                assert record['optimizer_step'] == i + 1 and record['actor_ids'] == [a['id'] for a in batch['actor']]
                assert all(record[k] == v for k, v in token['training_tokens'][str(seed)][i].items())
                parts = ['bit_loss', 'format_loss', 'actor_bit_loss', 'actor_format_loss']
                assert all(math.isfinite(record[k]) for k in [*parts, 'loss', 'gradient_norm']) and abs(record['loss'] - sum(record[k] for k in parts)) < 1e-5
                assert 0 < record['seconds'] < 15; train_times.append(record['seconds'])
            totals.append({'arm': arm, **{k: sum(r[k] for r in records) for k in ['query_tokens', 'actor_tokens']}})
        scores = {}
        for p, (obs, labels) in panels.items():
            raw = rows(folder / f'{p}.jsonl'); scores[p], elapsed = (standard_score if p == 'standard' else factor_score)(obs, labels, raw, 'binary')
            assert [r['prompt_tokens'] for r in raw] == token['eval_lengths'][p]; times.extend(elapsed); count += len(raw)
        actor, elapsed = oracle_score(folder)
        if kind != 'initial': times.extend(elapsed); count += 144
        factor_gate = all(c['both_correct'] == 8 for c in scores['factors']['cells']) and scores['factors']['invalid'] == 0
        summaries.append({'arm': arm, 'seed': seed, 'kind': kind, **scores, 'oracle': actor,
                          'factor_gate': factor_gate, 'strict_local_gate': scores['standard']['relation_gate_passed'] and factor_gate and actor['retained']})
    assert count == 7776 and len(train_times) == 576
    by_arm = {s['arm']: s for s in summaries}; contrasts = []
    for seed in [17, 29, 41]:
        a, b = by_arm[f'random{seed}'], by_arm[f'targeted{seed}']
        near = {k: sum(c['both_correct'] for c in x['factors']['cells'] if c['noise'] == 'near_miss') for k, x in [('random', a), ('targeted', b)]}
        standard_delta = b['standard']['paired_correct'] - a['standard']['paired_correct']
        contrasts.append({'seed': seed, 'near_miss_pairs': near, 'near_miss_pair_delta': near['targeted'] - near['random'],
                          'standard_pair_delta': standard_delta, 'actor_retained': b['oracle']['retained'],
                          'progress_diagnostic': near['targeted'] > near['random'] and standard_delta >= 0 and b['oracle']['retained']})
    result = {'summaries': summaries, 'matched_contrasts': contrasts, 'all_targeted_strict_local_gates': all(by_arm[f'targeted{s}']['strict_local_gate'] for s in [17, 29, 41]),
              'new_responses': count, 'reused_responses': 432, 'new_updates': 576, 'new_checkpoints': 6, 'token_counts': totals,
              'training_seconds': sum(train_times), 'max_update_seconds': max(train_times), 'generation_seconds': sum(times), 'max_generation_seconds': max(times),
              'errors': 0, 'autonomous_episodes': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({**result, 'summaries': [{**s, **{p: {k: v for k, v in s[p].items() if k not in ['cells', 'random_strata']} for p in panels}} for s in summaries]}))


if __name__ == '__main__': main()
