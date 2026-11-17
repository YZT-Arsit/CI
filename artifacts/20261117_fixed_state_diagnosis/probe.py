"""Free fixed-state inference; target labels are not read by this worker."""
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_action_binding'))
from learned_evidence import *
R = Path(__file__).resolve().parent


def main():
    plan = json.loads((R / 'PLAN.json').read_text())
    assert not (R / 'RUN_PLAN.json').exists()
    data = R / 'observations.jsonl'
    assert hashlib.sha256(data.read_bytes()).hexdigest() == plan['files'][data.name]
    cases = [json.loads(s) for s in data.read_text().splitlines()]
    assert len(cases) == plan['state_versions']
    initial = R.parent / '20261117_binding_factors/repair'
    checkpoint_seals = {arm: json.loads((initial / arm / 'CHECKPOINT_SEALED.json').read_text())
                        for arm in plan['arms']}
    run_plan = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'checkpoints': checkpoint_seals, 'read_labels': False,
                'own_estimates_only': True, 'model_updates': 0, 'batch_states': 2}
    (R / 'RUN_PLAN.json').write_text(json.dumps(run_plan, indent=2))
    for arm in plan['arms']:
        folder = initial / arm / 'adapter'
        assert all(hashlib.sha256((folder / p).read_bytes()).hexdigest() == h
                   for p, h in checkpoint_seals[arm]['adapter_hashes'].items())
        tokenizer, model = setup(17)
        set_peft_model_state_dict(model, load_file(str(folder / 'adapter_model.safetensors')))
        model.eval()
        pool = palette(tokenizer)['fresh']
        path = R / (arm + '.jsonl')
        assert not path.exists()
        for start in range(0, len(cases), 2):
            batch = cases[start:start + 2]
            texts, index = [], []
            estimates = [{} for _ in batch]
            records = [[] for _ in batch]
            nonces = [seed_for(row['base_group_id'], {'Planner': 0, 'Source': 1, 'Worker': 2}[row['observation']['role']]) for row in batch]
            for i, row in enumerate(batch):
                for action in row['observation']['actions']:
                    texts.append(query(tokenizer, row, action, pool, nonces[i])[0])
                    index.append((i, action))
            raws, seconds = generate(tokenizer, model, texts)
            for (i, action), text, raw in zip(index, texts, raws):
                matches = re.findall(r'ANSWER:\s*([01])\b', raw)
                valid = len(matches) == 1 and seconds < 9.5
                estimate = matches[0] if valid else 'unknown'
                estimates[i][action] = estimate
                records[i].append({'action': action, 'prompt': text, 'raw': raw,
                                   'valid': valid, 'estimate': estimate, 'seconds_batch': seconds})
            texts, orders = [], []
            for i, row in enumerate(batch):
                text, order = actor_text(row['observation'], pool, nonces[i], estimates[i])
                texts.append(text)
                orders.append(order)
            raws, seconds = generate(tokenizer, model, texts)
            with path.open('a', encoding='utf-8') as stream:
                for i, (row, text, raw, order) in enumerate(zip(batch, texts, raws, orders)):
                    matches = re.findall(r'ANSWER:\s*([ABC])\b', raw)
                    valid = len(matches) == 1 and ord(matches[0]) - 65 < len(order) and seconds < 9.5
                    selected = order[ord(matches[0]) - 65] if valid else None
                    stream.write(json.dumps({'case_id': row['case_id'], 'queries': records[i],
                                              'prompt': text, 'actions': order, 'raw': raw,
                                              'valid': valid, 'selected': selected,
                                              'seconds_batch': seconds}) + '\n')
            if (start + 2) % 100 == 0:
                print(json.dumps({'arm': arm, 'states_completed': start + len(batch)}), flush=True)
        del model
        gc.collect()
    seal = {'files': {arm + '.jsonl': hashlib.sha256((R / (arm + '.jsonl')).read_bytes()).hexdigest()
                      for arm in plan['arms']}, 'complete': True, 'labels_scored': False}
    (R / 'RESPONSES_SEALED.json').write_text(json.dumps(seal, indent=2))
    print('RESPONSES SEALED', flush=True)


if __name__ == '__main__':
    main()
