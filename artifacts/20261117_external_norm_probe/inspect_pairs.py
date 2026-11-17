"""Describe selected source-pair edits without treating them as causal interventions."""
import ast
import collections
import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parent


def main():
    labels = list(map(json.loads, (R / 'labels.jsonl').read_text().splitlines()))
    with (R / 'source/camber/privacylensplus.csv').open(encoding='utf-8', newline='') as stream:
        upstream = {row['example_id']: row for row in csv.DictReader(stream, delimiter='\t')}
    groups = collections.defaultdict(dict)
    for row in labels:
        groups[row['original_name']][row['label']] = ast.literal_eval(upstream[row['example_id']]['scenario'])
    deltas, totals = [], collections.Counter()
    for name, pair in groups.items():
        assert set(pair) == {'appropriate', 'inappropriate'}
        changed = [field for field in sorted(pair['appropriate']) if pair['appropriate'][field] != pair['inappropriate'][field]]
        assert changed
        totals.update(changed)
        deltas.append({'original_name': name, 'changed_fields': changed})
    result = {'selected_pairs': len(groups), 'field_change_counts': dict(totals), 'pairs': deltas,
              'interpretation': 'Synthetic positive vs negative seed edits; changes to subject or attribute mean these are not consent-only or task-equivalent counterfactuals.'}
    (R / 'PAIR_EDITS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}))


if __name__ == '__main__':
    main()
