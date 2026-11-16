import unittest
from kci.stratified import Trial,ComparisonPlan,FiniteRolloutPopulation
from kci.stream_cce import compare_rollouts


class StreamCCETests(unittest.TestCase):
    def test_known_equality_dominates_sampling_noise(self):
        pairs=[(Trial(True,True,1),Trial(False,False,1)),(Trial(False,False,1),Trial(True,True,1))]
        pop=FiniteRolloutPopulation(pairs,{'state':'one'})
        r=compare_rollouts(lambda seed:pairs[0],ComparisonPlan(n=256),{'state':'one'},upper_support=pop)
        self.assertEqual(r['status'],'UNKNOWN')
    def test_stream_success_and_cost(self):
        for b,level in [(Trial(True,False,3),'safe_success'),(Trial(True,True,14),'completion_cost')]:
            pair=(Trial(True,True,1),b);pop=FiniteRolloutPopulation([pair],{'state':'one'})
            r=compare_rollouts(lambda seed:pair,ComparisonPlan(),{'state':'one'},upper_support=pop)
            self.assertEqual(r['level'],level);self.assertEqual(r['chosen'],0)
    def test_without_upper_proof_abstains(self):
        pair=(Trial(True,True,1),Trial(True,True,14))
        r=compare_rollouts(lambda seed:pair,ComparisonPlan(),{})
        self.assertEqual(r['status'],'UNKNOWN')
    def test_wrong_certificate_binding_rejected(self):
        pair=(Trial(True,True,1),Trial(True,True,14));pop=FiniteRolloutPopulation([pair],{'state':'one'})
        with self.assertRaises(ValueError):compare_rollouts(lambda seed:pair,ComparisonPlan(),{'state':'two'},upper_support=pop)
    def test_support_drift_rejected(self):
        pair=(Trial(True,True,1),Trial(True,True,14));pop=FiniteRolloutPopulation([pair],{})
        with self.assertRaises(ValueError):compare_rollouts(lambda seed:(Trial(False,False,1),pair[1]),ComparisonPlan(),{},upper_support=pop)
    def test_unknown_not_dropped(self):
        r=compare_rollouts(lambda seed:(Trial(None,None,1),Trial(True,True,2)),ComparisonPlan(),{})
        self.assertEqual(r['status'],'UNKNOWN');self.assertEqual(len(r['sample_evidence']),1024)
    def test_cost_bound(self):
        with self.assertRaises(ValueError):compare_rollouts(lambda seed:(Trial(True,True,100),Trial(True,True,2)),ComparisonPlan(),{})


if __name__=='__main__':unittest.main()
