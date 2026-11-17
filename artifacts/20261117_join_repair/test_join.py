import unittest
import collections
from join_data import matched, training, fresh_standard, fresh_factors


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

    def test_schedule_balance_split_and_shared_controls(self):
        evaluation = {l['world']['group'] for l in fresh_standard()[1]} | {str(l['world']['seed']) for l in fresh_factors()[1]}
        training_ids = set()
        for seed in [17, 29, 41]:
            data, schedule = training(seed); self.assertEqual(len(schedule), 96)
            self.assertEqual(len(data['random']), 384); self.assertEqual(set(data['random']), set(data['targeted']))
            self.assertFalse(set(data['random']) & training_ids); training_ids.update(data['random'])
            permutations = collections.Counter()
            for b in schedule:
                self.assertEqual(sorted(b['labels']), [0, 0, 1, 1])
                self.assertEqual(sorted(e['group'] for e in b['actor']), ['positive', 'positive', 'zero', 'zero'])
                permutations.update(tuple(e['actions']) for e in b['actor'])
                for key in b['ids']:
                    a, c = data['random'][key], data['targeted'][key]
                    self.assertNotIn(a['group'], evaluation)
                    if b['stratum'][0] == 'control': self.assertEqual(a, c)
            self.assertEqual(len(permutations), 6); self.assertEqual(set(permutations.values()), {64})
        for generate, count in [(fresh_standard, 480), (fresh_factors, 288)]:
            obs, labels = generate(); self.assertEqual(len(obs), count); self.assertEqual(len(labels), count)
            self.assertEqual(sum(l['target'] for l in labels), count // 2)


if __name__ == '__main__': unittest.main()
