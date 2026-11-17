import itertools
import unittest
from graph_data import witnesses, truth


class WitnessTests(unittest.TestCase):
    def test_retention_direction_and_one_round(self):
        w = {'n': 3, 'frontier': [0], 'edges': [[0, 1], [1, 2]], 'target': 0}
        self.assertEqual(witnesses(w), {0})
        w['target'] = 1; self.assertEqual(witnesses(w), {0})
        w['target'] = 2; self.assertEqual(witnesses(w), set())
        w['edges'] = [[1, 0]]; w['target'] = 1; self.assertEqual(witnesses(w), set())

    def test_all_small_graphs_match_independent_matrix(self):
        edges = list(itertools.product(range(3), repeat=2))
        for mask in range(512):
            graph = [list(e) for i, e in enumerate(edges) if mask & (1 << i)]
            for start in range(8):
                for target in range(3):
                    w = {'n': 3, 'frontier': [i for i in range(3) if start & (1 << i)], 'edges': graph, 'target': target}
                    self.assertEqual(bool(witnesses(w)), bool(truth(w)))


if __name__ == '__main__': unittest.main()
