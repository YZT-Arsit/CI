"""Finite component scores and matched transitions, with no completion claim."""
import collections
import math
import re


def parse(raw, mode):
    if raw['seconds_batch'] >= 9.5: return None
    if mode == 'binary':
        matches = re.findall(r'ANSWER:\s*([01])\b', raw['raw'])
        return int(matches[0]) if len(matches) == 1 else None
    match = re.fullmatch(r'\s*PROOF:\s*([A-Za-z]+)\s*', raw['raw'])
    return match.group(1) if match else None


def score(obs, labels, raw, mode):
    assert mode in ['binary', 'proof', 'echo']; expected = 32 if mode == 'echo' else 288
    assert len(obs) == len(labels) == len(raw) == expected
    cells = collections.defaultdict(collections.Counter); pairs = collections.defaultdict(list); groups = collections.defaultdict(list)
    values = {}; times = []; total = collections.Counter(); lengths = collections.defaultdict(list)
    for i, (o, l, r) in enumerate(zip(obs, labels, raw)):
        assert o['id'] == l['id'] == r['id'] and math.isfinite(r['seconds_batch']) and r['seconds_batch'] > 0
        value = parse(r, mode); correct = value == l['target'] if mode == 'binary' else value in l['valid_witnesses']
        total.update(queries=1, correct=int(correct), invalid=int(value is None))
        key = (o['family'], o['known'], o['noise']); cells[key].update(queries=1, correct=int(correct), invalid=int(value is None))
        if mode != 'echo':
            total['positive_correct' if l['target'] else 'negative_correct'] += int(correct)
            pk = (o['seed'], o['known'], o['noise'], o['vocab']); pairs[pk].append((o['side'], correct, key))
            groups[o['seed']].append(correct); lengths[pk].append((r['prompt_tokens'], r['padded_width']))
            values[o['seed'], o['known'], o['noise'], o['vocab'], o['side']] = correct
        if i % 4 == 0:
            batch = raw[i:i + 4]; assert len({x['seconds_batch'] for x in batch}) == 1
            assert all(x['padded_width'] == max(y['prompt_tokens'] for y in batch) for x in batch); times.append(r['seconds_batch'])
    if mode == 'echo': return dict(total), times
    assert len(pairs) == 144 and len(groups) == 12 and all(len(v) == 24 for v in groups.values())
    assert len(values) == 288 and all(len(v) == 2 and len(set(v)) == 1 for v in lengths.values())
    for items in pairs.values():
        assert {x[0] for x in items} == {0, 1} and len({x[2] for x in items}) == 1
        cells[items[0][2]].update(pairs=1, both_correct=int(all(x[1] for x in items)))
    assert len(cells) == 18 and all(c['pairs'] == 8 for c in cells.values())
    contrasts = {}
    for name, field, a, b in [('add_irrelevant', 2, 'none', 'irrelevant'), ('add_near_miss', 2, 'none', 'near_miss'),
                              ('equal_count_near_miss', 2, 'irrelevant', 'near_miss'), ('expand_frontier', 1, 1, 4)]:
        counts = collections.Counter()
        for key, before in values.items():
            if key[field] != a: continue
            after_key = list(key); after_key[field] = b; after = values[tuple(after_key)]
            counts.update(cases=1, before_correct=int(before), after_correct=int(after))
            counts['correct_to_wrong' if before and not after else 'wrong_to_correct' if after and not before else 'unchanged_correctness'] += 1
        contrasts[name] = dict(counts)
    return {**dict(total), 'paired_correct': sum(c['both_correct'] for c in cells.values()), 'pairs': 144,
            'all_conditions_and_names_correct': sum(all(v) for v in groups.values()), 'base_groups': 12,
            'cells': [{'family': f, 'known': k, 'noise': n, **dict(v)} for (f, k, n), v in cells.items()],
            'matched_correctness_transitions': contrasts}, times
