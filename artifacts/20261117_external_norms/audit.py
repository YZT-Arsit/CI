"""Inventory immutable external examples without converting norms to training labels."""
import collections
import hashlib
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent
FIELDS = ['data_type', 'data_subject', 'data_sender', 'data_recipient', 'transmission_principle']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    manifest = json.loads((R / 'SOURCE_MANIFEST.json').read_text())
    card = (R / manifest['dataset_card_snapshot_path']).read_bytes()
    assert hashlib.sha256(card).hexdigest() == manifest['dataset_card_snapshot_sha256']
    for row in manifest['files']:
        raw = (R / row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == row['git_blob']
    data = json.loads((R / 'source/data/main_data.json').read_text(encoding='utf-8'))
    assert len(data) == len({row['name'] for row in data}) == 493
    source_counts = collections.Counter()
    details = collections.Counter()
    groups = collections.defaultdict(list)
    actions = collections.Counter()
    document_groups = collections.Counter()
    document_aliases = collections.defaultdict(set)
    index = []
    lengths = []
    for row in data:
        seed, vignette, trajectory = row['seed'], row['vignette'], row['trajectory']
        assert all(isinstance(seed[k], str) and seed[k].strip() for k in FIELDS)
        assert isinstance(vignette['story'], str) and vignette['story'].strip()
        assert isinstance(trajectory['executable_trajectory'], str) and trajectory['executable_trajectory'].strip()
        assert isinstance(trajectory['user_instruction'], str) and trajectory['user_instruction'].strip()
        assert isinstance(trajectory['toolkits'], list) and trajectory['toolkits']
        assert isinstance(trajectory['sensitive_info_items'], list) and trajectory['sensitive_info_items']
        group = digest({k: ' '.join(seed[k].lower().split()) for k in FIELDS})
        groups[group].append(row['name'])
        source_counts[seed['source']] += 1
        details[(seed['source'], bool(seed.get('source_details')))] += 1
        actions[trajectory['final_action']] += 1
        detail = seed.get('source_details', {})
        title = detail.get('regulation') or detail.get('paper')
        normalized_title = re.sub(r'[\W_]+', ' ', title.lower()).strip() if title else None
        document_group = digest({'source': seed['source'], 'title': normalized_title}) if title else None
        if document_group:
            key = (seed['source'], normalized_title, document_group)
            document_groups[key] += 1
            document_aliases[key].add(title)
        lengths.append(len(trajectory['executable_trajectory']))
        index.append({'original_name': row['name'], 'norm_group_sha256': group,
                      'source_kind': seed['source'], 'original_record_sha256': digest(row),
                      'source_document_group_sha256': document_group,
                      'assignment': 'EXTERNAL_EVALUATION_RESERVED', 'training_eligible': False})
    result = {'records': len(data), 'source_counts': dict(source_counts), 'unique_normalized_norm_groups': len(groups),
              'duplicate_norm_groups': {k: v for k, v in groups.items() if len(v) > 1},
              'source_details_presence': [{'source': a, 'present': b, 'records': n} for (a, b), n in details.items()],
              'final_action_types': dict(actions), 'trajectory_character_lengths': {'min': min(lengths), 'max': max(lengths)},
              'source_document_groups': [{'source': s, 'normalized_title': t, 'original_title_aliases': sorted(document_aliases[(s, t, h)]), 'group_sha256': h, 'records': n}
                                          for (s, t, h), n in document_groups.items()],
              'title_grouping_scope': 'Case, punctuation and whitespace normalization only; citation identity and semantic duplicates still require review before splitting.',
              'crowdsourcing_participant_provenance': 'Not present in released records; independence unresolved',
              'model_calls': 0, 'executed_workflows': 0, 'training_records_added': 0,
              'pretraining_contamination': 'UNKNOWN', 'data_license': manifest['dataset_license'],
              'scope': 'Source/schema inventory only. Negative-norm benchmark with simulated trajectories; no natural-business validity or transfer result.',
              'errors': 0, 'whole_goal_complete': False}
    (R / 'evaluation_index.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in index), encoding='utf-8')
    assert len(result['source_document_groups']) == len({g['group_sha256'] for g in result['source_document_groups']})
    result['evaluation_index_sha256'] = hashlib.sha256((R / 'evaluation_index.jsonl').read_bytes()).hexdigest()
    (R / 'AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['duplicate_norm_groups', 'final_action_types', 'source_document_groups']}))


if __name__ == '__main__':
    main()
