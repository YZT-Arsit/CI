from copy import deepcopy
from dataclasses import replace
import unittest
from experiments.task_evidence_dependency_probe import SOURCE
from kci.dynamic_tasks import DynamicTask
from kci.execution_envelope import VerifiedExecutionEnvelope
from kci.stratified import ComparisonPlan,Trial
from kci.stream_cce import compare_rollouts


class ExecutionEnvelopeTests(unittest.TestCase):
    def envelope(self,mode='safe_success',kind='meeting',horizon=12,cost_metric='steps'):
        env=DynamicTask(kind,SOURCE,horizon=horizon);state=env.decision_start()
        return VerifiedExecutionEnvelope(env,state,env.options(state),{'model':'test-frozen'},mode=mode,cost_metric=cost_metric)

    def test_every_canonical_and_malformed_proposal_is_contained(self):
        for kind in ('meeting','query'):
            e=self.envelope(kind=kind)
            candidates=['ABORT','SEND_CASE_LINK','FETCH_CITY_BUNDLE','not JSON',None,{'kind':'ABORT'},'x'*10000]
            for i,raw in enumerate(candidates):
                sample=e.sample_pair(lambda *args:raw,i)
                self.assertTrue(all(t.ci and t.success for t in e.validate_sample(sample,i)))
            self.assertTrue(any(step['blocked'] for record in e.samples.values() for side in record['execution'] for step in side['trace']))
    def test_callback_failure_is_recorded_as_blocked_proposal(self):
        e=self.envelope()
        def unavailable(*args):raise TimeoutError('model timed out')
        sample=e.sample_pair(unavailable,4)
        self.assertTrue(all(t.success for t in sample.trials))
        self.assertIn('proposal_error',e.samples[sample.evidence_id]['execution'][0]['trace'][0]['raw_proposal'])
    def test_ci_only_does_not_claim_equal_success(self):
        e=self.envelope('ci_only');c=e.upper_certificate()
        self.assertTrue(c['ci_equal']);self.assertFalse(c['safe_success_equal']);self.assertFalse(c['all_success'])
        sample=e.sample_pair(lambda obs,*args:'ABORT' if 'ABORT' in obs['actions'] else obs['actions'][0],3)
        self.assertTrue(all(t.ci for t in sample.trials));self.assertFalse(any(t.success for t in sample.trials))
    def test_infeasible_first_branch_cannot_get_success_proof(self):
        with self.assertRaises(ValueError):self.envelope(horizon=6)
    def test_unsafe_first_action_not_replaced(self):
        env=DynamicTask('meeting',SOURCE);s=env.decision_start();s,_=env.transition(s,'SEND_ROSTER')
        with self.assertRaises(ValueError):VerifiedExecutionEnvelope(env,s,env.options(s),{'model':'test'})
    def test_source_or_acl_drift_revokes_proof(self):
        e=self.envelope();e.env.source_acl['public_schedule'].clear()
        with self.assertRaises(ValueError):e.sample_pair(lambda *args:None,3)
    def test_graph_tamper_rejected(self):
        e=self.envelope();next(iter(e.nodes.values()))['viable']=False
        with self.assertRaises(ValueError):e.verify_current()
    def test_raw_trials_cannot_borrow_execution_certificate(self):
        e=self.envelope()
        with self.assertRaises(ValueError):compare_rollouts(lambda seed:(Trial(True,True,1),)*2,ComparisonPlan(n=2),e.binding,execution_envelope=e)
    def test_record_seed_and_content_bound(self):
        e=self.envelope();sample=e.sample_pair(lambda *args:None,3)
        with self.assertRaises(ValueError):e.validate_sample(sample,4)
        with self.assertRaises(ValueError):e.validate_sample(replace(sample,trials=(Trial(True,True,0),)*2),3)
        e.samples[sample.evidence_id]['seed']=99
        with self.assertRaises(ValueError):e.validate_sample(sample,3)
    def test_execution_certificate_allows_success_tier(self):
        e=self.envelope('ci_only')
        def policy(obs,seed,step,side):
            return 'ABORT' if side==1 and 'ABORT' in obs['actions'] else next(x for x in obs['actions'] if x!='ABORT')
        r=compare_rollouts(lambda seed:e.sample_pair(policy,seed),ComparisonPlan(n=32,cost_bound=20),e.binding,execution_envelope=e)
        self.assertEqual(r['level'],'safe_success');self.assertEqual(r['chosen'],0)
        self.assertFalse(r['upper_certificate']['raw_policy_risk_proven_zero'])
    def test_payload_cost_bound_checked(self):
        e=self.envelope(cost_metric='payload_bytes')
        with self.assertRaises(ValueError):compare_rollouts(lambda seed:e.sample_pair(lambda *args:None,seed),ComparisonPlan(n=2,cost_bound=1),e.binding,execution_envelope=e)
    def test_reference_binding_mismatch_rejected(self):
        e=self.envelope();binding=deepcopy(e.binding);binding['reference']['model']='different'
        with self.assertRaises(ValueError):e.verify_current(binding)
    def test_verified_cost_ranges_cover_executed_samples(self):
        e=self.envelope(cost_metric='payload_bytes');c=e.upper_certificate()
        lo,hi=c['paired_cost_difference_bounds']
        for i in range(8):
            s=e.sample_pair(lambda obs,seed,*args:obs['actions'][seed%len(obs['actions'])],i)
            self.assertLessEqual(lo,s.trials[1].cost-s.trials[0].cost)
            self.assertLessEqual(s.trials[1].cost-s.trials[0].cost,hi)
        e.cost_ranges=((0,0),(0,0))
        with self.assertRaises(ValueError):e.upper_certificate()
    def test_falsey_payload_types_are_not_silently_coerced(self):
        e=self.envelope();state=e.roots[0]
        raw={'actor':e.env.actor(state),'kind':'REQUEST_SANITIZED','recipient':0,'body':'','answer':'','fact_ids':[]}
        self.assertIsNone(e.decode_proposal(raw,state))
        raw['recipient']='';raw['fact_ids']={}
        self.assertIsNone(e.decode_proposal(raw,state))


if __name__=='__main__':unittest.main()
