"""Audit the secondary earlier-checkpoint reference without pooling the primary run."""
import hashlib
import json
from pathlib import Path
import replay

R = Path(__file__).resolve().parent


def main():
    plan = json.loads((R / 'reference/PLAN.json').read_text())
    assert hashlib.sha256((R / 'PLAN.json').read_bytes()).hexdigest() == plan['primary_plan_sha256']
    assert hashlib.sha256((R / 'reference/dev_worlds.json').read_bytes()).hexdigest() == plan['dev_sha256']
    assert (R / 'reference/dev_worlds.json').read_bytes() == (R / 'repair/dev_worlds.json').read_bytes()
    replay.run('reference', 'reference.py')
    root = R / 'reference/earlier_sft'
    def read(name):
        return list(map(json.loads, (root / (name + '.jsonl')).read_text().splitlines()))
    results = []
    for family in ['base', 'depth3', 'internal3', 'cycle4']:
        sets = [read(family + '_v' + str(view) + suffix) for view in (0, 1) for suffix in ['', '_swap']]
        assert len({len(rows) for rows in sets}) == 1
        assert [r['world_id'] for r in sets[0]] == [r['world_id'] for r in sets[2]]
        assert [r['world_id'] for r in sets[1]] == [r['world_id'] for r in sets[3]]
        results.append({'family': family, 'groups': len(sets[0]),
                        'all_four_strict_success': sum(all(row['strict_success'] for row in group) for group in zip(*sets))})
    for family in ['optional', 'authorized', 'draft', 'both']:
        a, b = read(family + '_v0'), read(family + '_v1')
        assert [r['world_id'] for r in a] == [r['world_id'] for r in b]
        results.append({'family': family, 'groups': len(a),
                        'both_views_strict_success': sum(x['strict_success'] and y['strict_success'] for x, y in zip(a, b)),
                        'both_views_valid_ci': sum(x['ci'] and not x['invalid'] and y['ci'] and not y['invalid'] for x, y in zip(a, b))})
    rows = [row for name in json.loads((R / 'reference/dev_worlds.json').read_text()) for row in read(name)]
    result = {'results': results, 'episodes': len(rows), 'strict_success': sum(row['strict_success'] for row in rows),
              'valid_ci': sum(row['ci'] and not row['invalid'] for row in rows), 'new_updates': 0,
              'scope': plan['scope'], 'whole_goal_complete': False}
    (R / 'reference/VIEW_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
