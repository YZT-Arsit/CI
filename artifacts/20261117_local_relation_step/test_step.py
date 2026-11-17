"""The one-step target must not silently use full transitive closure."""
import itertools
import json
import collections
import unittest
from step_data import reached_after_one, worlds, cases


class StepTests(unittest.TestCase):
    def test_direction_pair_holds_undirected_edges_and_degrees(self):
        train, dev, controls = worlds()
        self.assertEqual({a.get('family') for a, b in dev}, {'rewire', 'reverse_cycle'})
        for a, b in train + dev:
            if a['family'] != 'reverse_cycle': continue
            self.assertEqual(collections.Counter(tuple(sorted(e)) for e in a['edges']), collections.Counter(tuple(sorted(e)) for e in b['edges']))
            for column in [0, 1]:
                self.assertEqual(collections.Counter(e[column] for e in a['edges']), collections.Counter(e[column] for e in b['edges']))
            self.assertIn(a['target'], reached_after_one(a['frontier'], a['edges']))
            self.assertNotIn(b['target'], reached_after_one(b['frontier'], b['edges']))

    def test_frozen_json_roundtrip(self):
        for value in list(worlds()) + list(cases()):
            self.assertEqual(value, json.loads(json.dumps(value)))

    def test_direction_and_exact_round(self):
        self.assertEqual(reached_after_one(['a'], [('a', 'b'), ('b', 'c')]), {'a', 'b'})
        self.assertEqual(reached_after_one(['a'], [('b', 'a'), ('b', 'c')]), {'a'})
        self.assertEqual(reached_after_one(['b'], [('a', 'b'), ('b', 'c')]), {'b', 'c'})

    def test_exhaustive_against_boolean_matrix_round(self):
        nodes = range(3); potential = list(itertools.product(nodes, repeat=2))
        for edge_bits in itertools.product([0, 1], repeat=9):
            edges = [e for e, included in zip(potential, edge_bits) if included]
            matrix = [edge_bits[i * 3:i * 3 + 3] for i in nodes]
            for frontier_bits in itertools.product([0, 1], repeat=3):
                frontier = [i for i in nodes if frontier_bits[i]]
                expected = {j for j in nodes if frontier_bits[j] or sum(frontier_bits[i] * matrix[i][j] for i in nodes)}
                self.assertEqual(reached_after_one(frontier, edges), expected)


if __name__ == '__main__': unittest.main()
