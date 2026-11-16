import json
import unittest
from kci import (ToyCIEnv,ToyCandidateGenerator,JSONCandidateGenerator,ReferenceSearch,
                 SearchBudget,ContinuationEvaluator,certify_preference,SeededPolicy,
                 summarize_rollouts,outcome_from_trace)
from kci.contracts import Action,Trace,TraceStep,RefStatus,Verdict

class ExtraEdgeTests(unittest.TestCase):
    def setUp(self):
        self.env=ToyCIEnv()
        self.gen=ToyCandidateGenerator()
        self.ref=ReferenceSearch(self.env,self.gen)
        self.cce=ContinuationEvaluator(self.env,self.ref)
    def test_entity_similar_but_wrong_key_does_not_create_forbidden_join(self):
        s0=self.env.initial()
        s1,a1=self.env.transition(s0,self.env.act('A','SEND','R',('identity_other',)))
        self.assertEqual(a1.verdict,Verdict.VALID)
        s2,a2=self.env.transition(s1,self.env.act('B','SEND','R',('band',)))
        self.assertEqual(a2.verdict,Verdict.VALID)
        self.assertNotIn('person_band',self.env.known_to(s2,'R'))

    def test_reference_witness_not_automatically_optimal_for_llm_generator(self):
        def approximate_backend(prompt, n):
            p=json.loads(prompt)
            if p['role']=='R':
                return [{'actor':'R','kind':'FINALIZE','answer':'yes'}]
            if p['role']=='A' and p['awaiting_notification']:
                return [{'actor':'A','kind':'NOTIFY','recipient':'Alice','answer':'yes'}]
            return [{'actor':p['role'],'kind':'ABSTAIN'}]
        modelgen=JSONCandidateGenerator(approximate_backend)
        s0=self.env.initial()
        s1,_=self.env.transition(s0,self.env.act('A','SKIP'))
        s2,_=self.env.transition(s1,self.env.act('B','SEND','R',('band',)))
        result=ReferenceSearch(self.env,modelgen).search(s2)
        self.assertEqual(result.status,RefStatus.WITNESS_ONLY)
        self.assertEqual(result.best_success_cost,4.0)

    def test_no_next_action_from_mismatched_actor_cannot_leak(self):
        st=self.env.initial()
        act=self.env.act('B','SEND','R',('band',))
        ns,aud=self.env.transition(st,act)
        self.assertEqual(aud.verdict,Verdict.INVALID_ACTION)
        self.assertEqual(len(ns.receipts),0)

    def test_missing_provenance_do_not_generate_fake_pref(self):
        st=self.env.initial()
        untrusted=Action('A','SEND','R',('identity',),'Alice is linked to X17 and has band B2.')
        report=self.cce.evaluate(st,untrusted)
        self.assertEqual(report.immediate.verdict,Verdict.UNKNOWN)
        other=self.cce.evaluate(st,self.env.act('A','SKIP'))
        self.assertIsNone(certify_preference(other,report))

    def test_actual_policy_summary_distinguishes_success_violation_unknown(self):
        st=self.env.initial()
        next_st,va=self.env.transition(st,self.env.act('A','SEND','R',('identity',)))
        fail_st,vb=self.env.transition(next_st,self.env.act('B','SEND','R',('band',)))
        trace=Trace((TraceStep(st,self.env.act('A','SEND','R',('identity',)),va,next_st),
                     TraceStep(next_st,self.env.act('B','SEND','R',('band',)),vb,fail_st)))
        x=outcome_from_trace(trace)
        self.assertFalse(x.ci_compliant)
        stats=summarize_rollouts([trace],'demo_policy_v1')
        self.assertEqual(stats.confirmed_ci_violation_rate,1.0)
        self.assertEqual(stats.verified_safe_success_rate,0.0)

    def test_reference_proof_has_explicit_horizon(self):
        st=self.env.initial()
        rep=self.cce.evaluate(st,self.env.act('A','SKIP'))
        self.assertEqual(rep.reference.horizon,9)
        self.assertEqual(rep.reference.status,RefStatus.EXACT)

if __name__=='__main__':unittest.main()

class EmpiricalReferenceTests(unittest.TestCase):
    def test_fixed_reference_all_success_samples_do_not_prove_cost_preference(self):
        from kci import PairedReferenceComparator
        env=ToyCIEnv()
        def strong_policy(obs,seed):
            # Snapshot policy is blind to the oracle; it only sees authorized receipts.
            local={f.id for f in obs.local_facts}
            if obs.actor=='A':
                if obs.awaiting_notification:return env.act('A','NOTIFY','Alice',answer='yes')
                return env.act('A','SKIP')
            if obs.actor=='B':
                if 'eligible' in local:return env.act('B','SEND','R',('eligible',))
                if 'rule' in local:return env.act('B','COMPUTE_ELIGIBILITY')
                knows_identity=any('identity' in m.fact_ids for m in obs.authorized_receipts
                                   if m.recipient=='R')
                return (env.act('B','REQUEST_RULES',recipient='R') if knows_identity
                        else env.act('B','SEND','R',('band',)))
            if obs.pending_request:return env.act('R','SEND','B',('rule',))
            return env.act('R','FINALIZE',answer='yes')
        fixed=SeededPolicy(strong_policy,'strong_reference_frozen_v1')
        compare=PairedReferenceComparator(env,fixed,n=128,delta=.05)
        s=env.initial()
        result=compare.compare(s,env.act('A','SKIP'),env.act('A','SEND','R',('identity',)))
        self.assertEqual(result.status,'inconclusive')
        self.assertIsNone(result.preference)
        self.assertEqual(result.ci_diff_mean,0)
        self.assertEqual(result.success_diff_mean,0)
        self.assertEqual(result.cost_pairs,128)
        self.assertGreater(result.conditional_cost_diff_mean,0)

    def test_empirical_comparator_rejects_current_violation_as_future_comparison(self):
        from kci import PairedReferenceComparator
        env=ToyCIEnv()
        st,_=env.transition(env.initial(),env.act('A','SEND','R',('identity',)))
        policy=SeededPolicy(lambda obs,seed:env.act(obs.actor,'ABSTAIN'),'frozen_v1')
        cmp=PairedReferenceComparator(env,policy,n=10)
        result=cmp.compare(st,env.act('B','SEND','R',('band',)),env.act('B','REQUEST_RULES',recipient='R'))
        self.assertEqual(result.status,'ineligible')

class DataLeakageSplitTests(unittest.TestCase):
    def test_grouped_split_never_splits_counterfactual_variants(self):
        from kci import deterministic_group_split
        rows=[]
        for family in range(120):
            for history in range(3):
                rows.append({'metadata':{'scenario_group_id':f'world_{family}'},'variant':history})
        groups=deterministic_group_split(rows)
        seen={}
        for split,items in groups.items():
            for item in items:
                g=item['metadata']['scenario_group_id']
                if g in seen:self.assertEqual(seen[g],split)
                seen[g]=split
        self.assertEqual(len(seen),120)
        self.assertEqual(sum(len(v) for v in groups.values()),360)
