"""Strict copied-witness scoring and unchanged binary semantics, without inference."""
import collections
import re


def score(obs, labels, raw, mode):
    assert len(obs) == len(labels) == len(raw) == 480 and mode in ['binary', 'proof']
    cells = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(list); lengths = collections.defaultdict(list)
    random_groups = collections.defaultdict(list); random_strata = collections.defaultdict(collections.Counter)
    groups = collections.defaultdict(list); times = []; correct_total = invalid = 0
    for i, (o, label, r) in enumerate(zip(obs, labels, raw)):
        assert o['id'] == label['id'] == r['id']
        if mode == 'binary':
            matches = re.findall(r'ANSWER:\s*([01])\b', r['raw']); value = int(matches[0]) if len(matches) == 1 else None
        else:
            match = re.fullmatch(r'\s*PROOF:\s*([A-Za-z]+)\s*', r['raw']); value = match.group(1) if match else None
        if r['seconds_batch'] >= 9.5: value = None
        correct = value == label['target'] if mode == 'binary' else value in label['valid_witnesses']
        invalid += value is None; correct_total += correct
        key = (o['kind'], o['family'] if o['kind'] == 'switch' else None, o['noise'] if o['kind'] == 'switch' else None,
               o['vocab'], label['target'] if o['kind'] == 'random' else None)
        cells[key].update(queries=1, correct=int(correct), invalid=int(value is None))
        if o['kind'] == 'switch':
            pair_key = (o['group'], o['vocab'], o['view']); pairs[pair_key].append((o['side'], correct, key))
            lengths[pair_key].append((r['prompt_tokens'], r['padded_width']))
        elif o['kind'] == 'random':
            random_groups[o['group']].append(correct); world = label['world']
            random_strata[world['n'], len(world['edges']), len(world['frontier']), label['target'], o['vocab']].update(queries=1, correct=int(correct), invalid=int(value is None))
        if i % 4 == 0:
            batch = raw[i:i + 4]; assert len({x['seconds_batch'] for x in batch}) == 1
            assert all(x['padded_width'] == max(v['prompt_tokens'] for v in batch) for x in batch); times.append(r['seconds_batch'])
    assert len(pairs) == 144 and len(random_groups) == 32 and all(len(v) == 4 for v in random_groups.values())
    for (group, _, _), values in pairs.items():
        assert len(values) == 2 and {v[0] for v in values} == {0, 1} and values[0][2] == values[1][2]
        both = all(v[1] for v in values); cells[values[0][2]].update(pairs=1, both_correct=int(both)); groups[group].append(both)
    assert len(groups) == 36 and all(len(v) == 4 for v in groups.values())
    controls = [v for k, v in cells.items() if k[0] not in ['switch', 'random']]
    control_total = sum(s['queries'] for s in controls); control_correct = sum(s['correct'] for s in controls); assert control_total == 64
    switching = [v for k, v in cells.items() if k[0] == 'switch']; random_cells = [v for k, v in cells.items() if k[0] == 'random']
    assert len(switching) == 24 and all(s['pairs'] == 6 for s in switching) and len(random_cells) == 4 and all(s['queries'] == 32 for s in random_cells)
    equal = all(len(v) == 2 and len(set(v)) == 1 for v in lengths.values())
    random_all = sum(all(v) for v in random_groups.values())
    gate = all(s['both_correct'] / 6 >= .95 for s in switching) and control_correct == 64 and all(s['correct'] / 32 >= .95 for s in random_cells) and random_all >= 31 and invalid == 0 and equal
    result = {'queries': 480, 'correct': correct_total, 'invalid': invalid, 'changing_pairs': 144,
              'paired_correct': sum(s['both_correct'] for s in switching), 'all_four_groups': sum(all(v) for v in groups.values()), 'pair_groups': 36,
              'controls_correct': control_correct, 'controls': 64, 'random_correct': sum(s['correct'] for s in random_cells),
              'random_queries': 128, 'random_all_four_groups': random_all, 'random_groups': 32,
              'pair_token_control': equal, 'relation_gate_passed': gate,
              'random_strata': [{'nodes': n, 'edges': m, 'known_nodes': k, 'label': label, 'vocab': vocab, **dict(s)} for (n, m, k, label, vocab), s in random_strata.items()],
              'cells': [{'kind': k, 'family': f, 'noise': n, 'vocab': v, 'label': label, **dict(s)} for (k, f, n, v, label), s in cells.items()]}
    return result, times
