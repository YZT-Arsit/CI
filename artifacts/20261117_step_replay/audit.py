"""Separate conditional action repair from the unchanged full operator gate."""
import importlib.util
import json
import math
from replay_data import R, ROOT, PARENT, read, rows, sha, verify, replay_schedule, training_schedule, train_worlds, confirmation_cases


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


scorer = module('local_operator_scoring_replay', R.parent / '20261117_local_relation_step/audit_probe.py')
previous = module('prior_local_learning_audit', PARENT / 'audit.py')


def main():
    plan = verify(); seal = read(R / 'SEALED.json'); preflight = read(R / 'TOKEN_PREFLIGHT.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json') and all(sha(R / n) == h for n, h in seal['files'].items())
    assert preflight['plan_sha256'] == sha(R / 'PLAN.json') and preflight['script_sha256'] == sha(R / 'token_preflight.py')
    obs, labels = confirmation_cases(); assert obs == rows(R / 'observations.jsonl') and labels == rows(R / 'labels.jsonl')
    assert train_worlds() == read(R / 'training_worlds.json')
    summaries = []; times = []; training_times = []; response_count = 0; token_counts = []; initial_hash = None
    for arm in plan['arms']:
        target = R / 'study' / arm; trained = arm.startswith('replay'); initial = read(target / 'INITIAL_STATE.json')
        cp = read(target / 'CHECKPOINT_SEALED.json'); assert all(sha(target / 'adapter' / n) == h for n, h in cp['adapter_hashes'].items())
        if arm == 'initial': initial_hash = initial['trainable_sha256']
        if trained:
            assert initial['trainable_sha256'] == initial_hash and initial['checkpoint'] == plan['checkpoints']['initial']['path']
            assert cp['steps'] == 144 and cp['initial_adapter_hashes'] == plan['checkpoints']['initial']['seal']['adapter_hashes']
            seed = int(arm.removeprefix('replay')); records = rows(target / 'training.jsonl')
            local = training_schedule(seed); replay = replay_schedule(seed)
            assert local == rows(R / f'local_{seed}.jsonl') and replay == rows(R / f'replay_{seed}.jsonl')
            assert len(records) == 144 and not (target / 'TIMEOUT.json').exists()
            for i, (record, batch, rehearsal) in enumerate(zip(records, local, replay)):
                assert all(record[k] == v for k, v in batch.items()) and record['optimizer_step'] == i + 1
                assert record['replay_ids'] == [e['id'] for e in rehearsal['examples']] and record['good_indices'] == [e['good_indices'] for e in rehearsal['examples']]
                assert {k: record[k] for k in ['local_tokens', 'actor_tokens']} == preflight['training_tokens_by_step'][str(seed)][i]
                assert 0 < record['seconds'] < 15 and all(math.isfinite(record[k]) for k in ['loss', 'bit_loss', 'format_loss', 'actor_loss', 'actor_bit_loss', 'actor_format_loss', 'gradient_norm'])
                assert abs(record['actor_loss'] - record['actor_bit_loss'] - record['actor_format_loss']) < 1e-5
                assert abs(record['loss'] - record['bit_loss'] - record['format_loss'] - plan['actor_weight'] * record['actor_loss']) < 1e-5
                training_times.append(record['seconds'])
            token_counts.append({'arm': arm, 'local_tokens': sum(r['local_tokens'] for r in records), 'actor_tokens': sum(r['actor_tokens'] for r in records)})
        else:
            source = ROOT / plan['checkpoints'][arm]['path']
            assert cp == plan['checkpoints'][arm]['seal'] and initial['trainable_sha256'] == read(source / 'EVAL_IDENTITY.json')['before']
            assert (target / 'oracle.jsonl').read_bytes() == (source / 'oracle.jsonl').read_bytes()
            assert read(target / 'REUSED_ORACLE.json') == {'source': (source / 'oracle.jsonl').relative_to(ROOT).as_posix(), 'sha256': sha(source / 'oracle.jsonl'), 'new_calls': 0}
        identity = read(target / 'EVAL_IDENTITY.json'); assert identity['before'] == identity['after'] and identity['errors'] == 0
        result, elapsed = scorer.score(obs, labels, rows(target / 'operator.jsonl')); times.extend(elapsed); response_count += 320
        retained, elapsed = previous.oracle_score(target)
        if trained: times.extend(elapsed); response_count += 144
        summaries.append({'arm': arm, 'operator': result, 'oracle': retained, 'full_necessary_gate_passed': result['local_gate_passed'] and retained['retained']})
    by_arm = {r['arm']: r for r in summaries}; diagnostics = []
    for seed in plan['training_seeds']:
        baseline, replay = by_arm[f'local{seed}'], by_arm[f'replay{seed}']
        diagnostics.append({'seed': seed, 'paired_delta': replay['operator']['paired_correct'] - baseline['operator']['paired_correct'],
                            'action_retained': replay['oracle']['retained'],
                            'retained_and_no_pair_decline': replay['oracle']['retained'] and replay['operator']['paired_correct'] >= baseline['operator']['paired_correct']})
    assert response_count == 2672 and len(training_times) == 432
    result = {'summaries': summaries, 'matched_diagnostics': diagnostics,
              'all_replay_full_necessary_gates_passed': all(by_arm[f'replay{s}']['full_necessary_gate_passed'] for s in plan['training_seeds']),
              'new_responses': response_count, 'external_reused_responses': 576, 'new_updates': len(training_times),
              'training_seconds': sum(training_times), 'max_update_seconds': max(training_times), 'token_counts': token_counts,
              'generation_seconds': sum(times), 'max_generation_seconds': max(times), 'errors': 0,
              'autonomous_episodes': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({**result, 'summaries': [{**s, 'operator': {k: v for k, v in s['operator'].items() if k != 'cells'}} for s in summaries]}))


if __name__ == '__main__': main()
