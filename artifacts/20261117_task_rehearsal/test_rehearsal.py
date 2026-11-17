import collections
import unittest
from rehearsal_data import training, worlds, targets, check, prior, read


class RehearsalTest(unittest.TestCase):
    def test_supervision_is_matched_and_native(self):
        for seed in [17, 29, 41]:
            batches = training(seed)
            self.assertEqual(len(batches), 54)
            self.assertEqual(set(collections.Counter(tuple(b['global_cell']) for b in batches).values()), {3})
            for b in batches:
                for key in ['common', 'local', 'global']:
                    self.assertEqual(len(b[key]), 4)
                    self.assertEqual(sorted(e['label'] for e in b[key]), [0, 0, 1, 1])
                self.assertEqual([e['world_id'] for e in b['common']], [e['world_id'] for e in b['local']])
                self.assertTrue(all(a['prompt'] != c['prompt'] for a, c in zip(b['common'], b['local'])))
                self.assertEqual(len(b['actor']), 4)
                for e in b['global']:
                    self.assertEqual(check(e['case']), [])
                    self.assertEqual(e['label'], e['case']['candidates'][e['candidate_index']]['safe_completion'])
                    self.assertNotIn('Own estimates', e['prompt'])
            for start in range(0, 54, 18):
                self.assertEqual(len({tuple(b['global_cell']) for b in batches[start:start + 18]}), 18)

    def test_native_panel_is_unseen_and_actionable(self):
        ws, paired = worlds()
        self.assertEqual((len(ws), sum(map(len, ws.values())), len(paired)), (16, 32, 4))
        train_ids = {e['case']['base_group_id'] for b in training(17) for e in b['global']}
        self.assertFalse(train_ids & {w['world_id'] for v in ws.values() for w in v})
        for group in ws.values():
            for world in group:
                for case in targets(world):
                    self.assertEqual(check(case), [])


if __name__ == '__main__':
    unittest.main()
