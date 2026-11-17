import unittest
from factor_data import pair, dataset
from scoring_factors import score


def independent(w):
    matrix = [[int([i, j] in w['edges']) for j in range(16)] for i in range(16)]
    return int(w['target'] in w['frontier'] or any(matrix[i][w['target']] for i in w['frontier']))


class FactorTests(unittest.TestCase):
    def test_labels_and_frontier_expansion(self):
        for family in ['retention', 'direction', 'binding']:
            for seed in range(20):
                for noise in ['none', 'irrelevant', 'near_miss']:
                    small = pair(seed, family, 1, noise); large = pair(seed, family, 4, noise)
                    self.assertEqual([independent(w) for w in small], [1, 0])
                    self.assertEqual([independent(w) for w in large], [1, 0])
                    for a, b in zip(small, large):
                        self.assertEqual(a['edges'], b['edges'])
                        self.assertTrue(set(a['frontier']) < set(b['frontier']))
                        self.assertEqual(len(b['frontier']) - len(a['frontier']), 3)

    def test_controlled_pair_invariants(self):
        for family, core in [('retention', 0), ('direction', 1), ('binding', 2)]:
            for known in [1, 4]:
                for noise in ['none', 'irrelevant', 'near_miss']:
                    a, b = pair(71, family, known, noise)
                    self.assertEqual(a['target'], b['target'])
                    self.assertEqual(len(a['edges']), core + (0 if noise == 'none' else 12))
                    self.assertEqual(len(a['edges']), len(b['edges']))
                    if family == 'direction':
                        self.assertEqual(a['frontier'], b['frontier'])
                        self.assertEqual(a['edges'][0], b['edges'][0][::-1])
                        self.assertEqual(a['edges'][1:], b['edges'][1:])
                    else: self.assertEqual(a['edges'], b['edges'])
                    for w in [a, b]:
                        self.assertEqual(len(w['edges']), len(set(map(tuple, w['edges']))))
                        self.assertTrue(all(u != v for u, v in w['edges']))

    def test_frozen_counts_and_scoring_controls(self):
        data = dataset(); self.assertEqual(len(data['observations.jsonl']), 288)
        self.assertEqual(sum(l['target'] for l in data['labels.jsonl']), 144)
        for mode in ['binary', 'proof', 'echo']:
            prefix = 'echo_' if mode == 'echo' else ''; obs, labels = data[prefix + 'observations.jsonl'], data[prefix + 'labels.jsonl']
            raw = [{'id': o['id'], 'raw': 'ANSWER: ' + str(l['target']) if mode == 'binary' else 'PROOF: ' + l['valid_witnesses'][0],
                    'seconds_batch': .1, 'prompt_tokens': 100, 'padded_width': 100} for o, l in zip(obs, labels)]
            result, _ = score(obs, labels, raw, mode); self.assertEqual(result['correct'], len(obs))
            if mode != 'echo':
                self.assertEqual(result['paired_correct'], 144); self.assertEqual(result['all_conditions_and_names_correct'], 12)
                for r in raw: r['raw'] = 'ANSWER: 1' if mode == 'binary' else 'PROOF: none'
                result, _ = score(obs, labels, raw, mode); self.assertEqual(result['correct'], 144); self.assertEqual(result['paired_correct'], 0)
            raw[0]['id'] = 'wrong_identity'
            with self.assertRaises(AssertionError): score(obs, labels, raw, mode)


if __name__ == '__main__': unittest.main()
