import unittest
from copy import deepcopy
from dataclasses import replace

from kci import (ToyCIEnv,ToyCandidateGenerator,ReferenceSearch,
                 ContinuationEvaluator,DownstreamPolicyRepair,SeededPolicy,
                 PairedReferenceComparator,certify_preference)
from kci.contracts import Action,Trace,TraceStep,Verdict
from kci.rewards import Outcome,outcome_from_trace


def safe_policy(env,obs,seed):
    local={f.id for f in obs.local_facts}
    if obs.actor=='A':return env.act('A','NOTIFY','Alice',answer='yes')
    if obs.actor=='R':
        return (env.act('R','SEND','B',('rule',)) if obs.pending_request
                else env.act('R','FINALIZE',answer='yes'))
    if 'eligible' in local:return env.act('B','SEND','R',('eligible',))
    if 'rule' in local:return env.act('B','COMPUTE_ELIGIBILITY')
    identity=any('identity' in m.fact_ids and m.recipient=='R' for m in obs.authorized_receipts)
    return (env.act('B','REQUEST_RULES','R') if identity
            else env.act('B','SEND','R',('band',)))


class CoreRegressionTests(unittest.TestCase):
    def setUp(self):
        self.env=ToyCIEnv();self.gen=ToyCandidateGenerator()
        self.ref=ReferenceSearch(self.env,self.gen)
        self.cce=ContinuationEvaluator(self.env,self.ref)
        self.start=self.env.initial()

    def reports(self):
        return (self.cce.evaluate(self.start,self.env.act('A','SKIP')),
                self.cce.evaluate(self.start,self.env.act('A','SEND','R',('identity',))))

    def identity_step(self):
        action=self.env.act('A','SEND','R',('identity',))
        after,audit=self.env.transition(self.start,action)
        return TraceStep(self.start,action,audit,after)

    def test_cache_separates_accumulated_cost(self):
        state,_=self.env.transition(self.start,self.env.act('A','SKIP'))
        original=self.ref.search(state)
        changed=deepcopy(state);changed.cost+=100
        cached=self.ref.search(changed)
        fresh=ReferenceSearch(self.env,self.gen).search(changed)
        self.assertEqual(cached.best_success_cost,original.best_success_cost+100)
        self.assertEqual(cached.best_success_cost,fresh.best_success_cost)
        self.assertNotEqual(self.cce.state_id(state),self.cce.state_id(changed))

    def test_signature_preserves_event_identity_delivery_and_final_answer(self):
        state=self.identity_step().after
        for field,value in (('event_id','another_event'),('delivered',False)):
            changed=deepcopy(state)
            changed.receipts[0]=replace(changed.receipts[0],**{field:value})
            self.assertNotEqual(state.signature(),changed.signature())
        changed=deepcopy(state);changed.final_answer='yes'
        self.assertNotEqual(state.signature(),changed.signature())

    def test_comparison_rejects_different_authorized_prompt(self):
        a,b=self.reports()
        b=replace(b,observation=replace(b.observation,task=b.observation.task+' Extra context.'))
        with self.assertRaisesRegex(ValueError,'observation and prompt'):certify_preference(a,b)

    def test_cce_rejects_reference_for_another_environment_version(self):
        other=ToyCIEnv(purpose='another_authorized_purpose')
        with self.assertRaisesRegex(ValueError,'same environment version'):
            ContinuationEvaluator(self.env,ReferenceSearch(other,self.gen))

    def test_comparison_rejects_different_reference_context(self):
        a,b=self.reports()
        changes=(('reference_version','verifier_version','different_env'),
                 ('generator_version','generator_version','different_generator'),
                 ('search_budget','search_budget',(3000,12)),
                 ('horizon','horizon',10))
        for report_field,ref_field,value in changes:
            with self.subTest(field=report_field):
                changed=replace(b,**{report_field:value},reference=replace(b.reference,**{ref_field:value}))
                with self.assertRaisesRegex(ValueError,'same reference'):certify_preference(a,changed)

    def test_comparison_requires_matching_result_provenance(self):
        a,b=self.reports()
        with self.assertRaisesRegex(ValueError,'recorded comparison context'):
            certify_preference(a,replace(b,reference=replace(b.reference,generator_version='stale')))
        self.assertIsNone(certify_preference(a,replace(b,search_budget=None)))

    def test_exact_preference_keeps_reference_budget_and_versions(self):
        a,b=self.reports();pref=certify_preference(a,b)
        self.assertEqual(pref.source,'future_verified')
        self.assertEqual((pref.evidence['chosen_cost'],pref.evidence['rejected_cost']),(4.0,7.0))
        self.assertEqual(pref.evidence['reference_version'],self.env.version)
        self.assertEqual(pref.evidence['generator_version'],self.gen.version)
        self.assertEqual(pref.evidence['search_budget'],{'max_nodes':2000,'max_actions':12})

    def test_dpr_rejects_legal_partial_prefix(self):
        result=DownstreamPolicyRepair(self.cce,self.gen).repair(Trace((self.identity_step(),)))
        self.assertEqual(result.status,'unresolved');self.assertIsNone(result.preference)

    def test_dpr_does_not_turn_unknown_into_terminal_failure_label(self):
        first=self.identity_step();before=first.after
        action=Action('B','SEND','R',('time',),'unverified free text')
        after,audit=self.env.transition(before,action)
        stop=self.env.act('B','ABSTAIN');final,stop_audit=self.env.transition(after,stop)
        trace=Trace((first,TraceStep(before,action,audit,after),TraceStep(after,stop,stop_audit,final)))
        self.assertEqual(audit.verdict,Verdict.UNKNOWN);self.assertTrue(final.done)
        result=DownstreamPolicyRepair(self.cce,self.gen).repair(trace)
        self.assertEqual(result.status,'unresolved');self.assertIsNone(result.preference)

    def test_known_violation_disproves_safe_success_without_task_label(self):
        self.assertIs(Outcome(False,None,2.0,'stopped at violation').safe_success,False)
        first=self.identity_step();action=self.env.act('B','SEND','R',('band',))
        after,audit=self.env.transition(first.after,action)
        outcome=outcome_from_trace(Trace((first,TraceStep(first.after,action,audit,after))))
        self.assertIs(outcome.ci_compliant,False);self.assertIsNone(outcome.task_success)
        self.assertIs(outcome.safe_success,False)

    def test_unknown_after_violation_does_not_erase_confirmed_leak(self):
        first=self.identity_step();action=self.env.act('B','SEND','R',('band',))
        after,audit=self.env.transition(first.after,action)
        later=Action(after.actor,'FINALIZE',body='unverified free text',answer='yes')
        final,unknown=self.env.transition(after,later)
        trace=Trace((first,TraceStep(first.after,action,audit,after),TraceStep(after,later,unknown,final)))
        self.assertEqual(unknown.verdict,Verdict.UNKNOWN)
        outcome=outcome_from_trace(trace)
        self.assertIs(outcome.ci_compliant,False);self.assertIs(outcome.safe_success,False)


class PairedRegressionTests(unittest.TestCase):
    def setUp(self):
        self.env=ToyCIEnv();self.state=self.env.initial()
        self.a=self.env.act('A','SKIP')
        self.b=self.env.act('A','SEND','R',('identity',))

    def compare(self,policy,n=16):
        return PairedReferenceComparator(self.env,SeededPolicy(policy,'regression_policy_v1'),n=n).compare(
            self.state,self.a,self.b)

    def test_known_downstream_leaks_are_zero_safe_success_not_unknown(self):
        def policy(obs,seed):
            if obs.actor=='B':return self.env.act('B','SEND','R',('band',))
            return safe_policy(self.env,obs,seed)
        result=self.compare(policy)
        self.assertEqual(result.status,'estimated_preference')
        self.assertEqual(result.ci_diff_mean,1);self.assertEqual(result.success_diff_mean,1)
        self.assertEqual(result.success_known_pairs,16)
        self.assertEqual(result.preference.chosen.kind,'SKIP')
        self.assertTrue(all(row['b_task'] is None for row in result.sample_evidence))
        evidence=result.preference.evidence
        self.assertEqual(len(evidence['sample_seeds']),16)
        self.assertTrue(evidence['witness_actions'])
        self.assertEqual(evidence['environment_version'],self.env.version)

    def test_ci_advantage_without_safe_success_does_not_reward_refusal(self):
        def policy(obs,seed):
            identity=any('identity' in m.fact_ids and m.recipient=='R' for m in obs.authorized_receipts)
            return (self.env.act('B','SEND','R',('band',)) if identity
                    else self.env.act('B','ABSTAIN'))
        result=self.compare(policy)
        self.assertEqual(result.ci_diff_mean,1);self.assertEqual(result.success_diff_mean,0)
        self.assertIsNone(result.preference)
        self.assertIn('lacks a verified safe-success witness',result.reason)

    def test_finite_all_success_samples_do_not_certify_cost_preference(self):
        result=self.compare(lambda obs,seed:safe_policy(self.env,obs,seed))
        self.assertEqual(result.ci_diff_mean,0);self.assertEqual(result.success_diff_mean,0)
        self.assertEqual(result.conditional_cost_diff_mean,3.0)
        self.assertIsNone(result.conditional_cost_diff_radius)
        self.assertEqual(result.cost_pairs,16);self.assertIsNone(result.preference)

    def test_ci_priority_can_override_higher_safe_success_probability(self):
        def policy(obs,seed):
            local={f.id for f in obs.local_facts}
            if obs.actor=='B' and 'rule' not in local:
                identity=any('identity' in m.fact_ids and m.recipient=='R' for m in obs.authorized_receipts)
                if identity:
                    return (self.env.act('B','SEND','R',('band',)) if seed%4==0
                            else self.env.act('B','REQUEST_RULES','R'))
                return (self.env.act('B','SEND','R',('band',)) if seed%4==0
                        else self.env.act('B','ABSTAIN'))
            return safe_policy(self.env,obs,seed)
        result=self.compare(policy,n=256)
        self.assertGreater(result.ci_diff_mean,result.ci_diff_radius)
        self.assertLess(result.success_diff_mean,0)
        self.assertEqual(result.preference.chosen.kind,'SKIP')

    def test_unresolved_semantics_remain_unknown(self):
        result=self.compare(lambda obs,seed:Action(obs.actor,'SEND','R',('band',),'unverified text'))
        self.assertEqual(result.status,'unknown');self.assertIsNone(result.preference)
        self.assertIsNone(result.ci_diff_mean);self.assertEqual(result.ci_known_pairs,0)


if __name__=='__main__':unittest.main()
