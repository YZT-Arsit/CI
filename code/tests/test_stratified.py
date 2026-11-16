import unittest
from kci.stratified import ComparisonPlan, FiniteRolloutPopulation, Trial


class StratifiedTests(unittest.TestCase):
    def pop(self,a,b):return FiniteRolloutPopulation([(a,b)],{'test':'fixed-v1'})
    def test_equal_ci_success_advantage(self):
        p=self.pop(Trial(True,True,2),Trial(True,False,2))
        self.assertEqual(p.compare(ComparisonPlan(),use_upper_certificate=True)['level'],'safe_success')
        self.assertEqual(p.compare(ComparisonPlan())['status'],'UNKNOWN')
    def test_cost_only_after_proven_upper_equal(self):
        p=self.pop(Trial(True,True,2),Trial(True,True,14))
        self.assertEqual(p.compare(ComparisonPlan(),use_upper_certificate=True)['level'],'completion_cost')
    def test_task_advantage_cannot_override_ci(self):
        p=self.pop(Trial(False,True,1),Trial(True,True,15))
        r=p.compare(ComparisonPlan(),use_upper_certificate=True)
        self.assertEqual((r['level'],r['chosen']),('ci',1))
    def test_unknown_and_refusal(self):
        p=self.pop(Trial(None,None,1),Trial(True,True,3))
        self.assertEqual(p.compare(ComparisonPlan(),use_upper_certificate=True)['status'],'UNKNOWN')
        p=self.pop(Trial(True,False,1),Trial(False,True,3))
        self.assertEqual(p.compare(ComparisonPlan())['status'],'UNKNOWN')
    def test_bounds_and_identity(self):
        p=self.pop(Trial(True,True,17),Trial(True,True,1))
        with self.assertRaises(ValueError):p.compare(ComparisonPlan())
        with self.assertRaises(ValueError):ComparisonPlan(n=0)
        self.assertNotEqual(p.sha256,self.pop(Trial(True,True,16),Trial(True,True,1)).sha256)
    def test_equal_null(self):
        p=self.pop(Trial(True,True,1),Trial(True,True,1))
        self.assertEqual(p.compare(ComparisonPlan(),use_upper_certificate=True)['status'],'UNKNOWN')


if __name__=='__main__':unittest.main()
