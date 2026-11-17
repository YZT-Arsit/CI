import unittest
from factor_data import pair


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


if __name__ == '__main__': unittest.main()
