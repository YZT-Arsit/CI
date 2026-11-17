"""Reconstruct labels and autonomous transitions; retain every failed gate."""
import collections
import importlib.util
import json
import math
import sys
from rename_data import R, ROOT, PARENT, read, rows, sha, verify, batches, schedule, cases, third_cases, dev_cases, dev_worlds
from rename_views import views_for
import native_replay
BIT = R.parent / '20261117_bit_weight_repair'
sys.path.insert(0, str(BIT))
spec = importlib.util.spec_from_file_location('bit_audit_score', BIT / 'audit.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)


def panel(observations, labels, raw, dev=False):
    assert len(observations) == len(labels) == len(raw)
    stats = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(list)
    states = collections.defaultdict(list); times = []; groups = collections.defaultdict(list)
    for i, (o, label, answer) in enumerate(zip(observations, labels, raw)):
        assert o['id'] == label['id'] == answer['id']; bit = old.parse(answer, '01')
        correct = bit == str(label['target'])
        key = ((label['condition'].split('_v')[0], label['role']) if dev else (label['role'], label['depth']))
        stats[key].update(queries=1, correct=int(correct), invalid=int(bit is None), positive=int(label['target'] == 1),
                          false_positive=int(bit == '1' and label['target'] == 0), false_negative=int(bit == '0' and label['target'] == 1))
        if dev:
            c = label['condition']; state_key = (c, label['world_index'], label['state_index'])
            states[state_key].append(correct)
            if key[0] in ['base', 'depth3', 'depth4', 'internal3', 'cycle4']:
                pair_key = (c.removesuffix('_swap'), label['world_index'], label['state_index'], label['candidate_index'])
                pairs[pair_key].append((int(c.endswith('_swap')), label['target'], correct))
        elif label['changing']:
            pairs[o['group']].append((o['side'], label['target'], correct))
        if i % 4 == 0:
            assert len({v['seconds_batch'] for v in raw[i:i + 4]}) == 1; times.append(answer['seconds_batch'])
    changing = []
    for key, values in pairs.items():
        assert len(values) == 2 and {x[0] for x in values} == {0, 1}
        if {x[1] for x in values} == {0, 1}:
            changing.append(all(x[2] for x in values))
            if dev:
                c, wi, si, ci = key
                groups[c.split('_v')[0], wi, si, ci].append(all(x[2] for x in values))
    assert len(changing) == (288 if dev else 48)
    if dev: assert len(groups) == 144 and all(len(v) == 2 for v in groups.values())
    return {'queries': len(raw), 'correct': sum(v['correct'] for v in stats.values()),
            'invalid': sum(v['invalid'] for v in stats.values()), 'changing_pairs': len(changing),
            'both_correct': sum(changing), 'four_presentation_groups': len(groups),
            'all_four_correct': sum(all(v) for v in groups.values()),
            'states': len(states), 'all_correct_states': sum(all(v) for v in states.values()),
            'cells': [{'cell': list(k), **dict(v)} for k, v in stats.items()]}, times


def main():
    plan = verify(); seal = read(R / 'SEALED.json'); folder = R / 'study'
    assert seal['complete'] and seal['plan_sha256'] == sha(R / 'PLAN.json')
    assert all(sha(R / n) == h for n, h in seal['files'].items())
    assert plan == read(folder / 'PLAN.json') and plan['schedule'] == schedule()
    worlds, paired = dev_worlds(); assert worlds == read(R / 'dev_worlds.json') == read(folder / 'dev_worlds.json')
    assert paired == plan['paired_conditions']; bs = batches(); assert bs == rows(R / 'batches.jsonl')
    for prefix, (observations, labels) in [('', cases()), ('third_', third_cases()), ('dev_', dev_cases(worlds))]:
        assert observations == rows(R / (prefix + 'observations.jsonl')) and labels == rows(R / (prefix + 'labels.jsonl'))
    native_replay.run('study')
    updates = rows(folder / 'renamed/training.jsonl'); original = rows(PARENT / 'fit/joint/training.jsonl')
    assert len(updates) == len(original) == 144 and not (folder / 'renamed/TIMEOUT.json').exists()
    assert read(folder / 'renamed/INITIAL_STATE.json')['trainable_sha256'] == read(PARENT / 'fit/joint/INITIAL_STATE.json')['trainable_sha256']
    for step, (record, index) in enumerate(zip(updates, plan['schedule'])):
        assert record['step'] == record['optimizer_step'] == step + 1 and record['batch_index'] == index
        assert record['views'] == views_for(bs[index], step)
        assert record['tokens'] == original[step]['tokens']
        assert 0 < record['seconds'] < 15 and all(math.isfinite(record[k]) for k in ['loss', 'query_loss', 'actor_loss', 'gradient_norm'])
        assert abs(record['loss'] - record['query_loss'] - record['actor_loss']) < 1e-5
        pair = bs[index]['pairs'][str(bs[index]['mixed_depth'])]
        teacher = [{c['action']: str(c['safe_completion']) for c in row['candidates']} for row in pair]
        assert record['supplied'] == [teacher, teacher] == original[step]['supplied']
    results = []; times = []; response_count = 0
    for arm in plan['arms']:
        target = folder / arm; checkpoint = read(target / 'CHECKPOINT_SEALED.json')
        assert all(sha(target / 'adapter' / n) == h for n, h in checkpoint['adapter_hashes'].items())
        if arm == 'renamed':
            assert checkpoint['steps'] == 144 and checkpoint['initial_adapter_hashes'] == plan['checkpoint']['seal']['adapter_hashes']
        else:
            assert checkpoint == plan['fixed_checkpoint']['seal']
            for kind in ['query', 'oracle']:
                source = PARENT / 'fit/joint' / (kind + '_raw.jsonl')
                assert (target / (kind + '_raw.jsonl')).read_bytes() == source.read_bytes()
                assert read(target / ('REUSED_' + kind.upper() + '.json')) == {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source), 'new_calls': 0}
        identity = read(target / 'EVAL_IDENTITY.json'); assert identity['before'] == identity['after'] and identity['errors'] == 0
        old_score, old_times = old.score(target)
        if arm == 'renamed': times.extend(old_times); response_count += 480
        else: assert old_score == old.score(PARENT / 'fit/joint')[0]
        third, elapsed = panel(*third_cases(), rows(target / 'third_raw.jsonl')); times.extend(elapsed); response_count += 168
        dev, elapsed = panel(*dev_cases(worlds), rows(target / 'dev_raw.jsonl'), dev=True); times.extend(elapsed); response_count += 1536
        lengths = collections.defaultdict(list)
        for observations, raw in [(cases()[0], rows(target / 'query_raw.jsonl')), (third_cases()[0], rows(target / 'third_raw.jsonl'))]:
            for o, answer in zip(observations, raw): lengths[o['group']].append(answer['prompt_tokens'])
        assert len(lengths) == 84 and all(len(v) == 6 for v in lengths.values())
        token_control = all(len(set(v)) == 1 for v in lengths.values())
        records = {c: rows(target / (c + '.jsonl')) for c in worlds}; flat = [r for rs in records.values() for r in rs]; timing = {}
        for condition, rs in records.items():
            for index, row in enumerate(rs):
                for entry in row['trace']:
                    response_count += 1 + len(entry['queries']); key = (condition, index // 2, entry['phase'])
                    timing.setdefault((*key, 'actor'), set()).add(entry['seconds_batch'])
                    for q in entry['queries']: timing.setdefault((*key, 'query'), set()).add(q['seconds_batch'])
        assert all(len(v) == 1 for v in timing.values()); times.extend(next(iter(v)) for v in timing.values())
        families = ['base', 'depth3', 'depth4', 'internal3', 'cycle4']
        four = {f: sum(all(records[f + '_v' + str(v) + s][i]['strict_success'] for v in [0, 1] for s in ['', '_swap'])
                      for i in range(len(records[f + '_v0']))) for f in families}
        results.append({'arm': arm, 'train': old_score, 'third_palette_train': third, 'six_prompt_token_control': token_control,
                        'dev_fixed': dev, 'episodes': len(flat), 'strict_success': sum(r['strict_success'] for r in flat),
                        'violations': sum(not r['ci'] for r in flat), 'invalid_actions': sum(r['invalid'] for r in flat),
                        'invalid_queries': sum(r['invalid_queries'] for r in flat), 'all_four_groups': four,
                        'all_four_denominators': {f: len(records[f + '_v0']) for f in families}})
    fixed, renamed = results
    gate = (renamed['train']['paired_correct'] == {'original': 48, 'renamed': 48}
            and renamed['third_palette_train']['both_correct'] == 48 and renamed['six_prompt_token_control']
            and renamed['train']['oracle_retention_gate'] and renamed['strict_success'] >= fixed['strict_success']
            and renamed['violations'] == renamed['invalid_actions'] == renamed['invalid_queries'] == 0
            and all(renamed['all_four_groups'][f] > 0 for f in ['depth4', 'internal3', 'cycle4'])
            and renamed['dev_fixed']['both_correct'] > fixed['dev_fixed']['both_correct'])
    result = {'arms': results, 'local_gate_passed': gate, 'new_updates': 144, 'reused_responses': 480,
              'new_responses': response_count, 'new_fixed_panel_responses': 3888, 'full_autonomous_rollouts': 256,
              'training_seconds': sum(x['seconds'] for x in updates), 'max_update_seconds': max(x['seconds'] for x in updates),
              'generation_seconds': sum(times), 'max_generation_seconds': max(times),
              'errors': 0, 'whole_goal_complete': False, 'large_gpu': 'NO_GO'}
    (R / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({**result, 'arms': [{k: v for k, v in s.items() if k not in ['train', 'dev_fixed', 'third_palette_train']} for s in results]}))


if __name__ == '__main__':
    main()
