"""Native replay plus four-way presentation/history success for all four arms."""
import hashlib
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import replay


def main():
    spec = json.loads((R / 'PLAN.json').read_text())
    plan = json.loads((R / 'repair/PLAN.json').read_text())
    assert all(plan[k] == v for k, v in spec.items())
    assert all(hashlib.sha256((R / p).read_bytes()).hexdigest() == h for p, h in spec['source_hashes'].items())
    assert hashlib.sha256((R / 'repair/dev_worlds.json').read_bytes()).hexdigest() == spec['dev_sha256']
    old_root = R.parent / '20261117_binding_factors'
    assert hashlib.sha256((R.parent / '20261117_sampler_repair/PLAN.json').read_bytes()).hexdigest() == spec['prior_plan_sha256']
    assert hashlib.sha256((old_root / 'two_view_loss.py').read_bytes()).hexdigest() == spec['old_loss_sha256']
    assert all(hashlib.sha256((old_root / p).read_bytes()).hexdigest() == h for p, h in plan['unchanged_loss_dependencies'].items())
    replay.R = R
    replay.run('repair')
    def rows(arm, condition):
        return list(map(json.loads, (R / 'repair' / arm / (condition + '.jsonl')).read_text().splitlines()))
    results = []
    for arm in plan['arms']:
        for family in ['base', 'depth3', 'internal3', 'cycle4']:
            sets = [rows(arm, family + '_v' + str(view) + suffix) for view in (0, 1) for suffix in ['', '_swap']]
            assert len({len(items) for items in sets}) == 1
            robust = same = 0
            for a, b, c, d in zip(*sets):
                assert a['world_id'] == c['world_id'] and b['world_id'] == d['world_id']
                robust += all(r['strict_success'] for r in [a, b, c, d])
                for x, y in [(a, c), (b, d)]:
                    first, second = x['trace'][0], y['trace'][0]
                    same += first['format_valid'] and second['format_valid'] and first['selected'] == second['selected']
            results.append({'arm': arm, 'family': family, 'semantic_world_pairs': len(sets[0]),
                            'all_four_episodes_strict_success': robust, 'semantic_states': 2 * len(sets[0]),
                            'planner_same_valid_action_across_views': same})
        for family in ['optional', 'authorized', 'draft', 'both']:
            a, b = rows(arm, family + '_v0'), rows(arm, family + '_v1')
            assert [r['world_id'] for r in a] == [r['world_id'] for r in b]
            results.append({'arm': arm, 'family': family, 'semantic_worlds': len(a),
                            'both_views_strict_success': sum(x['strict_success'] and y['strict_success'] for x, y in zip(a, b)),
                            'both_views_valid_ci': sum(x['ci'] and not x['invalid'] and y['ci'] and not y['invalid'] for x, y in zip(a, b))})
    result = {'results': results, 'new_checkpoints': 2, 'reused_checkpoints': 2, 'new_optimizer_updates': 160,
              'scope': 'Single-seed four-arm loss comparison on same fresh DEV:36groups,56semanticworlds,112presentations perarm; no sourceholdout',
              'whole_goal_complete': False}
    (R / 'repair/VIEW_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
