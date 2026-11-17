"""Free query generation on original TRAIN inputs; label file never opened."""
from common import *
sys.path.insert(0, str(R.parent / '20261117_binding_factors'))
from render import query as train_query


def main():
    plan = verify()
    spec = read('TRAIN_QUERY_PLAN.json')
    assert sha(R / 'PLAN.json') == spec['parent_plan_sha256']
    assert all(sha(R / name) == digest for name, digest in spec['source_hashes'].items())
    assert sha(R / 'train_observations.jsonl') == spec['observations_sha256']
    assert not (R / 'TRAIN_QUERY_RUN_PLAN.json').exists()
    cases = rows('train_observations.jsonl')
    assert len(cases) == 320
    (R / 'TRAIN_QUERY_RUN_PLAN.json').write_text(json.dumps({'plan_sha256': sha(R / 'TRAIN_QUERY_PLAN.json'),
        'read_labels': False, 'model_updates': 0}, indent=2))
    paths = []
    for arm in spec['arms']:
        tokenizer, model = load(arm, plan)
        pool = palette(tokenizer)['train']
        path = R / f'train_queries_{arm}.jsonl'
        paths.append(path)
        for start in range(0, len(cases), plan['batch_states']):
            batch = cases[start:start + plan['batch_states']]
            texts, indices = [], []
            records = {case['case_id']: [] for case in batch}
            for case in batch:
                for action in case['observation']['actions']:
                    texts.append(train_query(case['observation'], pool, case['seeds'], action))
                    indices.append((case['case_id'], action))
            raws, seconds = generate(tokenizer, model, texts)
            for (cid, action), text, raw in zip(indices, texts, raws):
                found = re.findall(r'ANSWER:\s*([01])\b', raw)
                valid = len(found) == 1 and seconds < 9.5
                records[cid].append({'action': action, 'prompt': text, 'raw': raw, 'valid': valid,
                    'estimate': found[0] if valid else 'unknown', 'seconds_batch': seconds})
            for case in batch:
                append(path, {'case_id': case['case_id'], 'queries': records[case['case_id']]})
            if (start + len(batch)) % 80 == 0:
                print(json.dumps({'phase': 'train_query', 'arm': arm, 'states': start + len(batch)}), flush=True)
        del model
        gc.collect()
    seal('TRAIN_QUERIES_SEALED.json', paths, read_labels=False)


if __name__ == '__main__':
    main()
