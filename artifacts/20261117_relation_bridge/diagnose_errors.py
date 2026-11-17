"""Post-score descriptive cells; no model calls or gate changes."""
import collections
import json
from bridge_data import R, read, rows, sha, verify


def main():
    plan = verify(); seal = read(R / 'SEALED.json')
    assert seal['complete'] and all(sha(R / n) == h for n, h in seal['files'].items())
    assert read(R / 'RESULTS.json')['errors'] == 0
    labels = {x['id']: x for x in rows(R / 'labels.jsonl')}
    states = {x['id']: x for x in rows(R / 'states.jsonl')}
    reports = []
    for seed in plan['seeds']:
        for mode in plan['modes']:
            cells = collections.defaultdict(collections.Counter)
            for row in rows(R / 'study' / str(seed) / (mode + '_queries.jsonl')):
                label = labels[row['id']]; obs = label['case']['observation']
                action = label['case']['candidates'][label['candidate_index']]['action'].split(':')[0]
                names = ['condition:' + states[label['state_id']]['condition'], 'action:' + action,
                         'role:' + obs['role'], 'fact:' + str(label['relation_fact']),
                         'permission:' + str(obs['contract']['permission']),
                         'full_required:' + str(obs['contract']['binding_required'])]
                for name in names:
                    cells[name].update(n=1, correct=int(row['estimate'] == str(label['truth_target'])),
                                       predicted_one=int(row['estimate'] == '1'), target_one=label['truth_target'])
            reports.append({'seed': seed, 'mode': mode, 'cells': dict(cells)})
    result = {'post_score_descriptive': True, 'new_model_calls': 0, 'gate_changed': False,
              'source_sha256': sha(R / 'diagnose_errors.py'), 'seal_sha256': sha(R / 'SEALED.json'), 'summaries': reports}
    (R / 'ERROR_CELLS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for report in reports:
        if report['mode'] in ['direct', 'truth']:
            print(json.dumps({'seed': report['seed'], 'mode': report['mode'], 'cells': {k: v for k, v in report['cells'].items() if k.startswith(('action:', 'permission:', 'full_required:'))}}))


if __name__ == '__main__': main()
