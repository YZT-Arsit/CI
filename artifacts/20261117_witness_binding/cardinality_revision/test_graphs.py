import itertools
import unittest
from graph_data import witnesses, truth, paired, control, training, confirmation


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

    def test_counterfactuals_keep_declared_invariants(self):
        for family in ['edge_swap', 'frontier_swap', 'direction_cycle']:
            for seed in range(30):
                a, b = paired(seed, family, 20)
                self.assertEqual([truth(a), truth(b)], [1, 0]); self.assertEqual(len(a['frontier']), len(b['frontier']))
                if family == 'frontier_swap': self.assertEqual(a['edges'], b['edges'])
                else:
                    self.assertEqual(a['frontier'], b['frontier'])
                    for node in range(a['n']):
                        self.assertEqual(sum(e[0] == node for e in a['edges']), sum(e[0] == node for e in b['edges']))
                        self.assertEqual(sum(e[1] == node for e in a['edges']), sum(e[1] == node for e in b['edges']))
                if family == 'direction_cycle': self.assertEqual(sorted(sorted(e) for e in a['edges']), sorted(sorted(e) for e in b['edges']))
        for seed in range(20): self.assertGreaterEqual(len(witnesses(control(seed, 'multiple', 12))), 2)

    def test_training_balance_and_confirmation_sizes(self):
        for seed in [17, 29, 41]:
            worlds, schedule = training(seed); self.assertEqual(len(worlds), 768)
            self.assertTrue(all(sorted(r['labels']) == [0, 0, 1, 1] for r in schedule))
        obs, labels = confirmation(); self.assertEqual(len(obs), 480); self.assertEqual(sum(l['target'] for l in labels), 240)


if __name__ == '__main__': unittest.main()
