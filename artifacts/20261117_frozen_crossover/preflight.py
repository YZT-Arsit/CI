"""One real four-state inference batch before the frozen crossover worker."""
from common import *

plan = verify()
assert not (R / 'PREFLIGHT.json').exists() and not (R / 'RUN_PLAN.json').exists()
cases = rows('observations.jsonl')[:4]
tokenizer, model = load('teacher_sft', plan)
pool = palette(tokenizer)['fresh']
texts, indices = [], []
for case in cases:
    for action in case['observation']['actions']:
        texts.append(query(tokenizer, case, action, pool, nonce(case))[0])
        indices.append((case['case_id'], action))
raws, seconds = generate(tokenizer, model, texts)
assert seconds < 9.5
estimates = {case['case_id']: {} for case in cases}
query_records = []
for (cid, action), text, raw in zip(indices, texts, raws):
    found = re.findall(r'ANSWER:\s*([01])\b', raw)
    assert len(found) == 1
    estimates[cid][action] = found[0]
    query_records.append({'case_id': cid, 'action': action, 'prompt': text, 'raw': raw})
rendered = [actor_text(case['observation'], pool, nonce(case), estimates[case['case_id']]) for case in cases]
raws, actor_seconds = generate(tokenizer, model, [text for text, _ in rendered])
assert actor_seconds < 9.5
actors = [actor_record(case, text, order, raw, actor_seconds, estimates[case['case_id']])
          for case, (text, order), raw in zip(cases, rendered, raws)]
assert all(row['valid'] for row in actors)
result = {'arm': 'teacher_sft', 'states': 4, 'query_responses': len(texts), 'actor_responses': 4,
          'query_seconds': seconds, 'actor_seconds': actor_seconds, 'model_updates': 0,
          'read_labels': False, 'queries': query_records, 'actors': actors,
          'preflight_not_independent_evidence': True}
(R / 'PREFLIGHT.json').write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key not in ['queries', 'actors']}))
