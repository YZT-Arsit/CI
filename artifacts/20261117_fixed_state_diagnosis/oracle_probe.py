"""Teacher-estimate actor ceiling diagnostic; never learned-policy evidence."""
from probe import *


def main():
    plan = json.loads((R / 'PLAN.json').read_text())
    own = json.loads((R / 'RESPONSES_SEALED.json').read_text())
    assert own['complete'] and not own['labels_scored']
    assert all(hashlib.sha256((R / p).read_bytes()).hexdigest() == h for p, h in own['files'].items())
    assert not (R / 'ORACLE_RUN_PLAN.json').exists()
    assert hashlib.sha256((R / 'labels.jsonl').read_bytes()).hexdigest() == plan['files']['labels.jsonl']
    labels = {row['case_id']: row for row in map(json.loads, (R / 'labels.jsonl').read_text().splitlines())}
    cases = list(map(json.loads, (R / 'observations.jsonl').read_text().splitlines()))
    own_plan = json.loads((R / 'RUN_PLAN.json').read_text())
    (R / 'ORACLE_RUN_PLAN.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'read_labels': True, 'scope': 'ORACLE_ESTIMATE_ACTOR_DIAGNOSTIC_NOT_LEARNED_POLICY',
        'model_updates': 0, 'own_responses_sealed_before_oracle': own,
        'checkpoints': own_plan['checkpoints'], 'state_versions': len(cases)}, indent=2))
    for arm in plan['arms']:
        folder = R.parent / '20261117_binding_factors/repair' / arm / 'adapter'
        assert all(hashlib.sha256((folder / p).read_bytes()).hexdigest() == h
                   for p, h in own_plan['checkpoints'][arm]['adapter_hashes'].items())
        tokenizer, model = setup(17)
        set_peft_model_state_dict(model, load_file(str(folder / 'adapter_model.safetensors')))
        model.eval()
        pool = palette(tokenizer)['fresh']
        path = R / (arm + '_oracle.jsonl')
        assert not path.exists()
        for start in range(0, len(cases), 2):
            batch = cases[start:start + 2]
            texts, orders, estimates = [], [], []
            for row in batch:
                target = labels[row['case_id']]
                assert row['observation'] == target['observation']
                bits = {c['action']: str(c['safe_completion']) for c in target['candidates']}
                nonce = seed_for(row['base_group_id'], {'Planner': 0, 'Source': 1, 'Worker': 2}[row['observation']['role']])
                text, order = actor_text(row['observation'], pool, nonce, bits)
                texts.append(text)
                orders.append(order)
                estimates.append(bits)
            raws, seconds = generate(tokenizer, model, texts)
            with path.open('a', encoding='utf-8') as stream:
                for row, text, raw, order, bits in zip(batch, texts, raws, orders, estimates):
                    matches = re.findall(r'ANSWER:\s*([ABC])\b', raw)
                    valid = len(matches) == 1 and ord(matches[0]) - 65 < len(order) and seconds < 9.5
                    selected = order[ord(matches[0]) - 65] if valid else None
                    stream.write(json.dumps({'case_id': row['case_id'], 'prompt': text, 'actions': order,
                                              'oracle_estimates': bits, 'raw': raw, 'valid': valid,
                                              'selected': selected, 'seconds_batch': seconds}) + '\n')
            if (start + 2) % 100 == 0:
                print(json.dumps({'arm': arm, 'oracle_states_completed': start + len(batch)}), flush=True)
        del model
        gc.collect()
    (R / 'ORACLE_SEALED.json').write_text(json.dumps({
        'files': {arm + '_oracle.jsonl': hashlib.sha256((R / (arm + '_oracle.jsonl')).read_bytes()).hexdigest()
                  for arm in plan['arms']}, 'complete': True, 'labels_scored': False,
        'learned_policy_evidence': False}, indent=2))
    print('ORACLE RESPONSES SEALED', flush=True)


if __name__ == '__main__':
    main()
