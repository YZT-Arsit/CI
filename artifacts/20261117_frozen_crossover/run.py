"""Seal learned query predictions, then cross frozen actors; never open labels."""
from common import *


def main():
    plan = verify()
    assert not (R / 'RUN_PLAN.json').exists()
    assert sha(R / 'observations.jsonl') == plan['data_hashes']['observations.jsonl']
    cases = rows('observations.jsonl')
    (R / 'RUN_PLAN.json').write_text(json.dumps({'plan_sha256': sha(R / 'PLAN.json'),
        'read_labels': False, 'model_updates': 0, 'batch_states': plan['batch_states']}, indent=2))
    queries = {}
    paths = []
    for arm in plan['arms']:
        tokenizer, model = load(arm, plan)
        pool = palette(tokenizer)['fresh']
        path = R / f'queries_{arm}.jsonl'
        paths.append(path)
        queries[arm] = {}
        for start in range(0, len(cases), plan['batch_states']):
            batch = cases[start:start + plan['batch_states']]
            texts, indices = [], []
            records = {case['case_id']: [] for case in batch}
            for case in batch:
                for action in case['observation']['actions']:
                    texts.append(query(tokenizer, case, action, pool, nonce(case))[0])
                    indices.append((case['case_id'], action))
            raws, seconds = generate(tokenizer, model, texts)
            for (cid, action), text, raw in zip(indices, texts, raws):
                found = re.findall(r'ANSWER:\s*([01])\b', raw)
                valid = len(found) == 1 and seconds < 9.5
                records[cid].append({'action': action, 'prompt': text, 'raw': raw, 'valid': valid,
                                     'estimate': found[0] if valid else 'unknown', 'seconds_batch': seconds})
            for case in batch:
                cid = case['case_id']
                append(path, {'case_id': cid, 'queries': records[cid]})
                queries[arm][cid] = {q['action']: q['estimate'] for q in records[cid]}
            if (start + len(batch)) % 80 == 0:
                print(json.dumps({'phase': 'query', 'arm': arm, 'states': start + len(batch), 'seconds': seconds}), flush=True)
        del model
        gc.collect()
    seal('QUERIES_SEALED.json', paths)
    actor_paths = []
    for actor in plan['arms']:
        tokenizer, model = load(actor, plan)
        pool = palette(tokenizer)['fresh']
        for source in plan['arms']:
            path = R / f'actor_{actor}__query_{source}.jsonl'
            actor_paths.append(path)
            for start in range(0, len(cases), plan['batch_states']):
                batch = cases[start:start + plan['batch_states']]
                estimates = [queries[source][case['case_id']] for case in batch]
                rendered = [actor_text(case['observation'], pool, nonce(case), bits) for case, bits in zip(batch, estimates)]
                raws, seconds = generate(tokenizer, model, [text for text, _ in rendered])
                for case, (text, order), bits, raw in zip(batch, rendered, estimates, raws):
                    append(path, actor_record(case, text, order, raw, seconds, bits))
            print(json.dumps({'phase': 'actor', 'actor': actor, 'query': source, 'states': len(cases)}), flush=True)
        del model
        gc.collect()
    seal('LEARNED_SEALED.json', paths + actor_paths, learned_actor_responses=5040, read_labels=False)


if __name__ == '__main__':
    main()
