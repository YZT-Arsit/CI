"""Match proof supervision against binary training without rewriting failed gates."""
import importlib.util
import json
import math
from graph_data import R, ROOT, PARENT, NAMES, read, rows, sha, verify, training, confirmation, render
from scoring import score
spec = importlib.util.spec_from_file_location('previous_operator_audit_for_witness', R.parent / '20261117_step_learning/audit.py')
old_audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_audit)


def main():
    plan = verify(); seal = read(R / 'SEALED.json'); token = read(R / 'TOKEN_PREFLIGHT.json')
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json') and all(sha(R / n) == h for n, h in seal['files'].items())
    assert token['plan_sha256'] == sha(R / 'PLAN.json') and token['script_sha256'] == sha(R / 'token_preflight.py')
    obs, labels = confirmation(); assert obs == rows(R / 'observations.jsonl') and labels == rows(R / 'labels.jsonl')
    pool = read(NAMES / 'palettes.json')['train']; summaries = []; times = []; training_times = []; count = 0; initials = {}; tokens = []
    for arm, config in plan['arms'].items():
        seed, kind = config['seed'], config['kind']; target = R / 'study' / arm; start = read(target / 'INITIAL_STATE.json'); cp = read(target / 'CHECKPOINT_SEALED.json')
        assert all(sha(target / 'adapter' / n) == h for n, h in cp['adapter_hashes'].items())
        assert start['checkpoint'] == plan['checkpoints'][str(seed)]['path']
        if kind == 'start':
            initials[seed] = start['trainable_sha256']; source = ROOT / plan['checkpoints'][str(seed)]['path']
            assert cp == plan['checkpoints'][str(seed)]['seal'] and start['trainable_sha256'] == read(source / 'EVAL_IDENTITY.json')['before']
            assert (target / 'oracle.jsonl').read_bytes() == (source / 'oracle.jsonl').read_bytes()
            assert read(target / 'REUSED_ORACLE.json') == {'source': (source / 'oracle.jsonl').relative_to(ROOT).as_posix(), 'sha256': sha(source / 'oracle.jsonl'), 'new_calls': 0}
        else:
            assert start['trainable_sha256'] == initials[seed] and cp['steps'] == 192 and cp['initial_adapter_hashes'] == plan['checkpoints'][str(seed)]['seal']['adapter_hashes']
            data, schedule = training(seed); records = rows(target / 'training.jsonl')
            assert data == read(R / f'worlds_{seed}.json') and schedule == rows(R / f'schedule_{seed}.jsonl') and len(records) == 192 and not (target / 'TIMEOUT.json').exists()
            for i, (r, b) in enumerate(zip(records, schedule)):
                assert all(r[k] == b[k] for k in ['step', 'stratum', 'ids', 'names', 'labels', 'replay_source_step']) and r['optimizer_step'] == i + 1
                assert r['actor_ids'] == [e['id'] for e in b['actor']]
                assert r['valid_witnesses'] == [render(data[key], pool, name)[1] for key, name in zip(b['ids'], b['names'])]
                expected = token['training_tokens_by_step'][str(seed)][i]
                assert r['query_tokens'] == expected['query_tokens'] and r['actor_tokens'] == expected['actor_tokens']
                assert r['proof_tokens'] == (expected['proof_tokens'] if kind == 'proof' else 0)
                parts = ['bit_loss', 'format_loss', 'actor_bit_loss', 'actor_format_loss']
                if kind == 'proof': parts += ['proof_bit_loss', 'proof_format_loss']
                else: assert r['proof_bit_loss'] is None and r['proof_format_loss'] is None
                assert all(math.isfinite(r[k]) for k in parts + ['loss', 'gradient_norm']) and abs(r['loss'] - sum(r[k] for k in parts)) < 1e-5
                assert 0 < r['seconds'] < 15; training_times.append(r['seconds'])
            tokens.append({'arm': arm, **{k: sum(r[k] for r in records) for k in ['query_tokens', 'actor_tokens', 'proof_tokens']}})
        identity = read(target / 'EVAL_IDENTITY.json'); assert identity['before'] == identity['after'] and identity['errors'] == 0
        panels = {}
        for mode in ['binary', 'proof']:
            panels[mode], elapsed = score(obs, labels, rows(target / (mode + '.jsonl')), mode); times.extend(elapsed); count += 480
        actor, elapsed = old_audit.oracle_score(target)
        if kind != 'start': times.extend(elapsed); count += 144
        summaries.append({'arm': arm, 'seed': seed, 'kind': kind, **panels, 'oracle': actor,
                          'binary_and_action_gate': panels['binary']['relation_gate_passed'] and actor['retained'],
                          'witness_joint_necessary_gate': all(panels[m]['relation_gate_passed'] for m in ['binary', 'proof']) and actor['retained']})
    assert count == 9504 and len(training_times) == 1152
    by_arm = {r['arm']: r for r in summaries}
    contrasts = [{'seed': s, 'binary_pair_delta': by_arm[f'proof{s}']['binary']['paired_correct'] - by_arm[f'bit{s}']['binary']['paired_correct'],
                  'binary_random_delta': by_arm[f'proof{s}']['binary']['random_correct'] - by_arm[f'bit{s}']['binary']['random_correct'],
                  'valid_witness_delta': by_arm[f'proof{s}']['proof']['correct'] - by_arm[f'bit{s}']['proof']['correct'],
                  'proof_arm_actor_retained': by_arm[f'proof{s}']['oracle']['retained']} for s in [17, 29, 41]]
    result = {'summaries': summaries, 'matched_contrasts': contrasts, 'all_witness_joint_gates_passed': all(by_arm[f'proof{s}']['witness_joint_necessary_gate'] for s in [17, 29, 41]),
              'new_responses': count, 'external_reused_responses': 432, 'new_updates': len(training_times), 'token_counts': tokens,
              'training_seconds': sum(training_times), 'max_update_seconds': max(training_times), 'generation_seconds': sum(times), 'max_generation_seconds': max(times),
              'errors': 0, 'autonomous_episodes': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({**result, 'summaries': [{**s, **{m: {k: v for k, v in s[m].items() if k not in ['cells', 'random_strata']} for m in ['binary', 'proof']}} for s in summaries]}))


if __name__ == '__main__': main()
