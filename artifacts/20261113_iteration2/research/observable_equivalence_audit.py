"""Corpus-wide exact-prompt checks, including action values and strict preferences."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(text):
    return sha256(text.encode('utf-8')).hexdigest()


def direction(a, b):
    """Strict evidence contract, not a new reward or a sampled preference."""
    safe_a = a['minimum_safe_total_cost'] is not None
    safe_b = b['minimum_safe_total_cost'] is not None
    if a['immediate'] == 'valid' and safe_a and b['immediate'] == 'violation':
        return 1
    if b['immediate'] == 'valid' and safe_b and a['immediate'] == 'violation':
        return -1
    if a['immediate'] != 'valid' or b['immediate'] != 'valid':
        return None
    exact = {'exact_within_horizon', 'proven_unreachable_within_horizon'}
    if a['reference_status'] not in exact or b['reference_status'] not in exact:
        return None
    if safe_a != safe_b:
        return 1 if safe_a else -1
    if not safe_a:
        return 0
    delta = b['minimum_safe_total_cost'] - a['minimum_safe_total_cost']
    return 1 if delta > 0 else -1 if delta < 0 else 0


def run(input_dir, output_dir):
    started = time.perf_counter()
    files = sorted(Path(input_dir).glob('*.json'))
    groups, preferences, hashes = defaultdict(list), [], {}
    case_count = state_count = 0
    for file in files:
        raw = file.read_bytes()
        record = json.loads(raw)
        if 'case_spec' not in record or 'states' not in record:
            continue
        hashes[file.name] = sha256(raw).hexdigest()
        assert record['enumeration_complete'], 'cannot certify an incompletely enumerated case'
        case_count += 1
        for state in record['states']:
            state_count += 1
            prompt_id = digest(state['prompt'])
            groups[prompt_id].append({'case_id': record['case_spec']['case_id'], 'state_id': state['state_id'],
                'role': json.loads(state['prompt'])['role'], 'prefix': state['prefix'],
                'actions': {canonical(a['action']): a for a in state['actions']}})
        preferences.extend(record['preferences'])
    conflicts = {'menu': [], 'immediate': [], 'safe_reachability': [], 'minimum_total_cost': []}
    for prompt_id, members in groups.items():
        if len(members) < 2:
            continue
        menus = {tuple(sorted(m['actions'])) for m in members}
        if len(menus) > 1:
            conflicts['menu'].append({'prompt_sha256': prompt_id, 'members': len(members)})
        shared_actions = set.intersection(*(set(m['actions']) for m in members))
        for action in shared_actions:
            reports = [m['actions'][action] for m in members]
            signatures = {
                'immediate': [a['immediate'] for a in reports],
                'safe_reachability': [a['minimum_safe_total_cost'] is not None for a in reports],
                'minimum_total_cost': [a['minimum_safe_total_cost'] for a in reports]
            }
            for field, values in signatures.items():
                if len(set(values)) > 1:
                    examples = []
                    for value in dict.fromkeys(values):
                        i = values.index(value)
                        examples.append({'value': value, 'case_id': members[i]['case_id'],
                                         'state_id': members[i]['state_id'], 'prefix': members[i]['prefix']})
                    conflicts[field].append({'prompt_sha256': prompt_id, 'role': members[0]['role'],
                        'action': json.loads(action), 'examples': examples})
    inconsistent, unique = {}, {}
    raw_ids = defaultdict(list)
    for row in preferences:
        signature = digest('\n'.join(row[k] for k in ('prompt', 'chosen', 'rejected')))
        raw_ids[signature].append(row['metadata'].get('raw_record_id', digest(canonical(row))))
    for row in preferences:
        signature = digest('\n'.join(row[k] for k in ('prompt', 'chosen', 'rejected')))
        if signature in unique:
            continue
        unique[signature] = row
        prompt_id = digest(row['prompt'])
        chosen, rejected = canonical(json.loads(row['chosen'])), canonical(json.loads(row['rejected']))
        members = groups.get(prompt_id, [])
        results = []
        for member in members:
            available = chosen in member['actions'] and rejected in member['actions']
            relation = direction(member['actions'][chosen], member['actions'][rejected]) if available else None
            results.append({'relation': relation, 'case_id': member['case_id'], 'state_id': member['state_id']})
        if not results or any(r['relation'] != 1 for r in results):
            reasons = []
            if not results:
                reasons.append('no_matching_observation_state')
            if any(r['relation'] == -1 for r in results):
                reasons.append('reverse_strict_preference_in_same_prompt_class')
            if any(r['relation'] == 0 for r in results):
                reasons.append('tie_in_same_prompt_class_no_uniform_strict_preference')
            if any(r['relation'] is None for r in results):
                reasons.append('missing_action_or_insufficient_evidence_in_same_prompt_class')
            inconsistent[signature] = {'preference_signature': signature, 'prompt_sha256': prompt_id,
                'raw_record_ids': sorted(set(raw_ids[signature])), 'quarantine_reasons': reasons,
                'role': json.loads(row['prompt'])['role'], 'chosen': json.loads(chosen), 'rejected': json.loads(rejected),
                'compatible_state_results': results}
    output = Path(output_dir)
    output.mkdir(exist_ok=False, parents=True)
    (output/'conflicts.json').write_text(json.dumps(conflicts, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'inconsistent_preferences.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in inconsistent.values()), encoding='utf-8')
    summary = {'input_case_files': case_count, 'enumerated_nonterminal_states': state_count,
        'exact_prompt_classes': len(groups), 'multi_state_prompt_classes': sum(len(x)>1 for x in groups.values()),
        'raw_preferences': len(preferences), 'unique_prompt_preferences': len(unique),
        'conflicting_observation_action_counts': {k: len(v) for k,v in conflicts.items()},
        'strict_preferences_not_uniformly_supported': len(inconsistent),
        'raw_records_recommended_for_quarantine': sum(len(r['raw_record_ids']) for r in inconsistent.values()),
        'strict_preferences_with_actual_reverse': sum(any(x['relation']==-1 for x in r['compatible_state_results']) for r in inconsistent.values()),
        'input_sha256': hashes, 'source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'exact prompt equality across enumerated declared cases only; future values reuse production reference and require independent oracle audit',
        'rule': 'A hard strict preference must retain direction +1 in every enumerated state with the same full training prompt.',
        'seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.input, a.output)
