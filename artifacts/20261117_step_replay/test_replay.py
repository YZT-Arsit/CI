"""Rehearsal must preserve ties, balance labels, and cover option orders."""
import collections
import itertools
import unittest
from replay_data import replay_schedule, worker_pairs


class ReplayTests(unittest.TestCase):
    def test_balanced_ties_and_option_permutations(self):
        for seed in [17, 29, 41]:
            records = replay_schedule(seed); self.assertEqual(len(records), 144)
            orders = collections.Counter()
            for batch in records:
                self.assertEqual(len(batch['examples']), 4)
                self.assertEqual(collections.Counter(e['group'] for e in batch['examples']), {'zero': 2, 'positive': 2})
                for e in batch['examples']:
                    expected = {'PUBLISH_MINIMAL', 'WITHHOLD'} if e['group'] == 'zero' else {'PUBLISH_FULL'}
                    self.assertEqual({e['actions'][i] for i in e['good_indices']}, expected)
                    orders[e['group'], tuple(e['actions'])] += 1
            self.assertEqual(len(orders), 12)
            self.assertEqual(set(orders.values()), {48})

    def test_schedule_is_reproducible_and_sources_are_train(self):
        a = replay_schedule(17); self.assertTrue(a); self.assertEqual(a, replay_schedule(17)); self.assertNotEqual(a, replay_schedule(29))
        allowed = {r['state_id'] for p in worker_pairs() for r in p}
        self.assertEqual({e['state_id'] for b in a for e in b['examples']}, allowed)


if __name__ == '__main__': unittest.main()
