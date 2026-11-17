import unittest
from join_data import matched


def label(w):
    matrix = [[int([i, j] in w['edges']) for j in range(w['n'])] for i in range(w['n'])]
    return int(w['target'] in w['frontier'] or any(matrix[i][w['target']] for i in w['frontier']))


class JoinTests(unittest.TestCase):
    def test_matched_labels_and_cores(self):
        for family in ['edge_swap', 'frontier_swap', 'direction_cycle']:
            for known in [1, 4]:
                for seed in range(8):
                    a, b = matched(seed * 1000, family, known)
                    self.assertEqual([label(w) for w in a], [1, 0])
                    self.assertEqual([label(w) for w in b], [1, 0])
                    for x, y in zip(a, b):
                        self.assertEqual(x['n'], 16); self.assertEqual(len(x['frontier']), known)
                        for key in ['id', 'group', 'side', 'n', 'frontier', 'target', 'family']:
                            self.assertEqual(x[key], y[key])
                        self.assertEqual(x['edges'][:-12], y['edges'][:-12]); self.assertEqual(len(x['edges']), len(y['edges']))
                        self.assertEqual(len(y['edges']), len(set(map(tuple, y['edges']))))
                    for worlds in [a, b]: self.assertEqual(worlds[0]['edges'][-12:], worlds[1]['edges'][-12:])
                    union = set(b[0]['frontier'] + b[1]['frontier']); target = b[0]['target']
                    extra = b[0]['edges'][-12:]
                    self.assertEqual(sum(u in union and v != target for u, v in extra), 6)
                    self.assertEqual(sum(u not in union and v == target for u, v in extra), 6)


if __name__ == '__main__': unittest.main()
