"""Read-only diversity/overlap audit of the original synthetic CPU corpus.

Normalization is a descriptive surface-pattern diagnostic. It neither creates
new CI labels nor treats normalized preference pairs as independent tasks.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re


BASE_WORLD_GROUP = 'toy_eligibility_join_v1'
NAME_RULES = (
    (r'\bK1\d{3}\b', 'CURRENT_KEY'),
    (r'\bK2\d{3}\b', 'OTHER_KEY'),
    (r'\bPerson\d{3}\b', 'PERSON'),
)
BAND_RULE = (r'\bB[123]\b', 'ELIGIBLE_BAND')


def normalize(value, collapse_eligible_bands=False):
    for pattern, replacement in NAME_RULES:
        value = re.sub(pattern, replacement, value)
    if collapse_eligible_bands:
        value = re.sub(*BAND_RULE, value)
    return value


def pair_key(record, mode='exact'):
    values = tuple(record[key] for key in ('prompt', 'chosen', 'rejected'))
    if mode == 'exact':
        return values
    return tuple(normalize(value, mode == 'names_and_eligible_bands') for value in values)


def audit(files):
    records = []
    file_hashes = []
    for path in files:
        data = path.read_bytes()
        records.extend(json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip())
        file_hashes.append({'path': str(path.resolve()), 'sha256': sha256(data).hexdigest()})
    exact_sources = defaultdict(set)
    raw_sources = Counter()
    rendering_indices = set()
    rule_families = Counter()
    chosen_answers = Counter()
    pattern_renderings = {mode: defaultdict(set) for mode in ('names', 'names_and_eligible_bands')}
    for record in records:
        meta = record['metadata']
        index = meta['world_config']['index']
        sources = meta['training_sources']
        rendering_indices.add(index)
        rule_families[meta['rule_family_id']] += 1
        raw_sources.update(sources)
        exact_sources[pair_key(record)].update(sources)
        chosen_answers[json.loads(record['chosen'])['answer']] += 1
        for mode, patterns in pattern_renderings.items():
            patterns[pair_key(record, mode)].add(index)
    unique_sources = Counter(source for sources in exact_sources.values() for source in sources)
    membership = Counter('+'.join(sorted(sources)) for sources in exact_sources.values())
    patterns = {}
    for mode, mapping in pattern_renderings.items():
        patterns[mode] = {
            'unique_preference_patterns': len(mapping),
            'patterns_in_multiple_surface_renderings': sum(len(indices) > 1 for indices in mapping.values()),
            'surface_rendering_counts_per_pattern': dict(Counter(len(indices) for indices in mapping.values())),
            'pattern_groups': [
                {'pattern_sha256': sha256(json.dumps(key, ensure_ascii=False).encode('utf-8')).hexdigest(),
                 'surface_rendering_indices': sorted(indices)}
                for key, indices in sorted(mapping.items())
            ],
        }
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'evidence_class': 'synthetic_corpus_descriptive_audit_NOT_independent_task_count',
        'files': file_hashes, 'raw_records': len(records), 'exact_unique_preferences': len(exact_sources),
        'exact_duplicate_rows': len(records) - len(exact_sources),
        'surface_renderings': len(rendering_indices), 'surface_rendering_indices': sorted(rendering_indices),
        'base_world_group': BASE_WORLD_GROUP, 'base_world_groups': 1,
        'rule_family_record_counts': dict(rule_families),
        'raw_source_counts': dict(raw_sources), 'exact_unique_source_counts': dict(unique_sources),
        'unique_source_memberships': dict(membership), 'chosen_answer_record_counts': dict(chosen_answers),
        'normalization': {'name_rules': list(NAME_RULES), 'eligible_band_rule': BAND_RULE,
                          'scope': 'Apply to prompt, chosen, rejected strings only; no labels are changed.',
                          'band_assumption': 'In this corpus B1/B2/B3 are all eligible under the same fixed rule.',
                          'warning': 'Normalized pairs are not independent tasks or new world/rule families.'},
        'normalized_patterns': patterns,
        'primary_leakage_safe_split': {
            'group_key': BASE_WORLD_GROUP, 'independent_holdout_available': False,
            'training_evaluation_ready': False,
            'reason': 'All entity/name/band renderings instantiate the same base world and relation family. '
                      'An index-based split is a leakage demonstration or in-template sanity check only.',
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Directory containing worlds_*.jsonl shards')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    files = sorted(Path(args.input).glob('worlds_*.jsonl'))
    if not files:
        raise SystemExit('No synthetic shards found')
    result = audit(files)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('raw_records', 'exact_unique_preferences',
          'surface_renderings', 'base_world_groups', 'unique_source_memberships')}, ensure_ascii=False))
    print(json.dumps({key: value['unique_preference_patterns']
                     for key, value in result['normalized_patterns'].items()}))
