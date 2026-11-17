import collections
import importlib.util
import json
import math
from join_fit_data import F, R, ROOT, data, read, rows, sha, verified
spec = importlib.util.spec_from_file_location('factor_parser_for_join_fit', data.FACTORS / 'scoring_factors.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def main():
    parent, plan = verified(); seal = read(F / 'SEALED.json'); pseal = read(R / 'SEALED.json')
    assert pseal['complete'] and pseal['plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(R / name) == digest for name, digest in pseal['files'].items())
    assert seal['complete'] and seal['parent_seal_sha256'] == sha(R / 'SEALED.json') and seal['plan_sha256'] == sha(F / 'PLAN.json')
    assert all(sha(F / name) == digest for name, digest in seal['files'].items())
    obs, labels = rows(F / 'observations.jsonl'), rows(F / 'labels.jsonl'); summaries = []; times = []; lengths_by_arm = {}
    for arm, cfg in plan['arms'].items():
        source = R / 'study' / arm; identity = read(F / 'outputs' / (arm + '_IDENTITY.json'))
        assert identity['errors'] == 0 and identity['before'] == identity['after'] == read(source / 'EVAL_IDENTITY.json')['before']
        assert identity['checkpoint'] == {'path': source.relative_to(ROOT).as_posix(), 'seal': read(source / 'CHECKPOINT_SEALED.json')}
        selected = [(o, l) for o, l in zip(obs, labels) if o['seed'] == cfg['seed']]; raw = rows(F / 'outputs' / (arm + '.jsonl'))
        assert len(raw) == len(selected) == 56; counts = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(list)
        lengths_by_arm[arm] = [(v['prompt_tokens'], v['padded_width']) for v in raw]
        for i, ((o, l), v) in enumerate(zip(selected, raw)):
            assert o['id'] == l['id'] == v['id'] and math.isfinite(v['seconds_batch']) and 0 < v['seconds_batch']
            value = module.parse(v, 'binary'); correct = value == l['target']; key = o['background']
            counts[key].update(queries=1, correct=int(correct), invalid=int(value is None))
            counts[key]['positive_correct' if l['target'] else 'negative_correct'] += int(correct)
            if o['kind'] == 'switch': pairs[key, o['group']].append((o['side'], correct, v['prompt_tokens'], v['padded_width']))
            if i % 4 == 0:
                batch = raw[i:i + 4]; assert len({x['seconds_batch'] for x in batch}) == 1
                assert all(x['padded_width'] == max(y['prompt_tokens'] for y in batch) for x in batch); times.append(v['seconds_batch'])
        assert counts['shared']['queries'] == 8 and len(pairs) == 24
        for (key, group), items in pairs.items():
            assert len(items) == 2 and {v[0] for v in items} == {0, 1} and len({tuple(v[2:]) for v in items}) == 1
            counts[key].update(pairs=1, paired_correct=int(all(v[1] for v in items)))
        assert all(counts[k]['queries'] == 24 and counts[k]['pairs'] == 12 for k in ['random', 'targeted'])
        summaries.append({'arm': arm, 'seed': cfg['seed'], 'trained_background': cfg['kind'], 'scores': {k: dict(v) for k, v in counts.items()}})
    assert all(lengths_by_arm['random' + str(s)] == lengths_by_arm['targeted' + str(s)] for s in [17, 29, 41])
    result = {'summaries': summaries, 'new_responses': 336, 'new_updates': 0, 'new_checkpoints': 0, 'reused_responses': 0,
              'generation_seconds': sum(times), 'max_generation_seconds': max(times), 'errors': 0,
              'selected_train_only': True, 'whole_goal_complete': False, 'autonomous_episodes': 0, 'large_gpu': 'NO_GO'}
    (F / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__': main()
