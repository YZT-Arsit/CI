"""Supplementary all-query confusion counts; no change to fit gates or model calls."""
import collections
import json
import re
from repair_data import R, PARENT, read, rows, sha, verify, cases


def main():
    verify(); seal = read(R / 'SEALED.json'); registration = read(R / 'ERROR_DIRECTION_PLAN.json')
    assert seal['complete'] and all(sha(R / n) == h for n, h in seal['files'].items())
    assert sha(R / 'audit_error_direction.py') == registration['source_sha256']
    assert sha(R / 'PLAN.json') == registration['main_plan_sha256']
    obs, labels = cases(); summaries = []
    for arm, folder in [('joint', PARENT / 'fit/joint'), ('uniform_joint', R / 'fit')]:
        raw = rows(folder / 'query_raw.jsonl'); assert len(raw) == len(labels) == 336
        stats = collections.defaultdict(collections.Counter)
        for o, label, answer in zip(obs, labels, raw):
            assert o['id'] == label['id'] == answer['id']
            found = re.findall(r'ANSWER:\s*([01])\b', answer['raw'])
            bit = int(found[0]) if len(found) == 1 and answer['seconds_batch'] < 9.5 else None
            key = (o['vocab'], label['role'], label['depth'], label['changing'])
            stats[key].update(queries=1, positive=label['target'], negative=1-label['target'],
                              false_positive=int(bit == 1 and label['target'] == 0),
                              false_negative=int(bit == 0 and label['target'] == 1), invalid=int(bit is None))
        summaries.append({'arm': arm, 'raw_sha256': sha(folder / 'query_raw.jsonl'),
                          'cells': [{'vocab': v, 'role': role, 'depth': depth, 'changing': changing, **dict(s)}
                                    for (v, role, depth, changing), s in stats.items()]})
    result = {'summaries': summaries, 'errors': 0, 'model_calls': 0, 'primary_gate_changed': False,
              'scope': registration['scope'], 'main_seal_sha256': sha(R / 'SEALED.json')}
    (R / 'ERROR_DIRECTION_RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'errors': 0, 'scored_query_responses_including_control_reuse': 672, 'model_calls': 0}))


if __name__ == '__main__':
    main()
