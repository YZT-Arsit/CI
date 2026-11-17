"""Regression: depth coverage must include value-changing relationship pairs."""
import collections
import unittest
from contrast_data import schedule,matched_groups


class CoverageTests(unittest.TestCase):
    def test_relation_contrasts_in_every_role_depth_cell(self):
        batches=schedule();counts=collections.Counter();roles=collections.Counter()
        for b in batches:
            role=b['key'][0];d=b['mixed_depth'];roles[role]+=1
            a,c=b['pairs'][str(d)]
            counts[role,d]+=any(x['safe_completion']!=y['safe_completion'] for x,y in zip(a['candidates'],c['candidates']))
        self.assertEqual(len(batches),96)
        self.assertEqual(dict(roles),{'Planner':32,'Source':32,'Worker':32})
        for role in roles:
            for depth in [1,2,3]:self.assertGreaterEqual(counts[role,depth],6,(role,depth,dict(counts)))

    def test_matched_labels_and_all_anchor_conditions(self):
        batches=schedule();self.assertEqual({tuple(b['key']) for b in batches},set(matched_groups()))
        for b in batches:
            sig=lambda d:[[(c['immediate_safe'],c['safe_completion'],c['value']) for c in r['candidates']] for r in b['pairs'][str(d)]]
            self.assertEqual(sig(1),sig(2));self.assertEqual(sig(2),sig(3));self.assertEqual(len(b['views']),2)


if __name__=='__main__':unittest.main()
