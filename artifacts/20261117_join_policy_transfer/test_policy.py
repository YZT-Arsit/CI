import unittest
from policy_data import worlds, targets, environment, check


class PolicyDataTest(unittest.TestCase):
    def test_native_labels_and_shared_views(self):
        panel, paired = worlds(); self.assertEqual(sum(map(len, panel.values())), 32)
        for condition, cases in panel.items():
            for world in cases:
                self.assertFalse([error for case in targets(world) for error in check(case)])
                view = condition.replace('_v0', '_v1')
                if '_v0' in condition:
                    counterpart = next(w for w in panel[view] if w['world_id'] == world['world_id'])
                    self.assertEqual({k: v for k, v in world.items() if k != 'base_group_id'}, {k: v for k, v in counterpart.items() if k != 'base_group_id'})
        for condition in paired:
            for a, b in zip(panel[condition], panel[condition + '_swap']):
                self.assertEqual(a['base_group_id'], b['base_group_id'])
                best = []
                for world in [a, b]:
                    env = environment(world); obs = env.observation(env.initial()); case = next(c for c in targets(world) if c['observation'] == obs)
                    best.append({case['candidates'][i]['action'] for i in case['optimal_indices']})
                self.assertFalse(best[0] & best[1])


if __name__ == '__main__': unittest.main()
