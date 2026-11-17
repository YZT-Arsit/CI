"""Separate oracle upper bounds and exact TRAIN fit, after learned sealing."""
from common import *
sys.path.insert(0, str(R.parent / '20261117_binding_factors'))
from render import actor as training_actor


def main():
    plan = verify()
    learned = read('LEARNED_SEALED.json')
    assert learned['complete'] and not learned['read_labels']
    assert all(sha(R / name) == digest for name, digest in learned['files'].items())
    assert not (R / 'ORACLE_RUN_PLAN.json').exists()
    assert all(sha(R / name) == digest for name, digest in plan['data_hashes'].items())
    cases = rows('observations.jsonl')
    labels = {row['case_id']: row for row in rows('labels.jsonl')}
    train = rows('train_oracle.jsonl')
    (R / 'ORACLE_RUN_PLAN.json').write_text(json.dumps({'plan_sha256': sha(R / 'PLAN.json'),
        'learned_seal_sha256': sha(R / 'LEARNED_SEALED.json'), 'read_labels': True,
        'learned_policy_evidence': False, 'model_updates': 0}, indent=2))
    paths = []
    for arm in plan['arms']:
        tokenizer, model = load(arm, plan)
        pools = palette(tokenizer)
        for split, data in [('dev', cases)] + ([('train', train)] if arm in plan['train_arms'] else []):
            path = R / f'oracle_{split}_{arm}.jsonl'
            paths.append(path)
            for start in range(0, len(data), plan['batch_states']):
                batch = data[start:start + plan['batch_states']]
                texts, orders, estimates = [], [], []
                for case in batch:
                    target = labels[case['case_id']] if split == 'dev' else case
                    bits = {c['action']: str(c['safe_completion']) for c in target['candidates']}
                    if split == 'dev':
                        text, order = actor_text(case['observation'], pools['fresh'], nonce(case), bits)
                    else:
                        text, order = training_actor(case['observation'], pools['train'], case['seeds'], bits)
                    texts.append(text); orders.append(order); estimates.append(bits)
                raws, seconds = generate(tokenizer, model, texts)
                for case, text, order, bits, raw in zip(batch, texts, orders, estimates, raws):
                    append(path, actor_record(case, text, order, raw, seconds, bits))
            print(json.dumps({'phase': 'oracle', 'arm': arm, 'split': split, 'states': len(data)}), flush=True)
        del model
        gc.collect()
    seal('ORACLE_SEALED.json', paths, learned_policy_evidence=False)


if __name__ == '__main__':
    main()
