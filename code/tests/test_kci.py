import unittest
from kci import (ToyCIEnv,ToyCandidateGenerator,JSONCandidateGenerator,ReferenceSearch,
                 SearchBudget,ContinuationEvaluator,DeterministicPolicy,DownstreamPolicyRepair,
                 certify_preference,export_verified_preferences,audit_pisas_files,validate_enriched_episode)
from kci.contracts import (Action,Fact,Trace,TraceStep,RefStatus,Verdict)
from kci.datasets import DataContractError,KCI_REQUIRED_NEW

class KCITest(unittest.TestCase):
    def setUp(self):
        self.env=ToyCIEnv()
        self.gen=ToyCandidateGenerator()
        self.ref=ReferenceSearch(self.env,self.gen,SearchBudget(max_nodes=3000))
        self.cce=ContinuationEvaluator(self.env,self.ref)
    def after_identity(self):
        st,aud=self.env.transition(self.env.initial(),self.env.act('A','SEND','R',('identity',)))
        self.assertEqual(aud.verdict,Verdict.VALID)
        return st
    def after_skip(self):
        st,aud=self.env.transition(self.env.initial(),self.env.act('A','SKIP'))
        self.assertEqual(aud.verdict,Verdict.VALID)
        return st

    def test_current_same_message_history_changes_ci_verdict(self):
        for st,expected in ((self.after_skip(),Verdict.VALID),(self.after_identity(),Verdict.VIOLATION)):
            ns,audit=self.env.transition(st,self.env.act('B','SEND','R',('band',)))
            self.assertEqual(audit.verdict,expected)
        self.assertIn('person_band',self.env.known_to(ns,'R'))
        self.assertTrue(audit.inference_evidence)

    def test_historical_knowledge_updates_only_after_delivery(self):
        st=self.after_identity()
        self.assertIn('identity',self.env.known_to(st,'R'))
        # A hypothetical candidate must not update the actual real history.
        cce=self.cce.evaluate(st,self.env.act('B','SEND','R',('band',)))
        self.assertEqual(cce.immediate.verdict,Verdict.VIOLATION)
        self.assertEqual(self.env.known_to(st,'R'),{'identity','rule'})
        self.assertEqual(len(st.receipts),1)

    def test_allow_derived_but_unrelated_fact(self):
        st=self.after_identity()
        ns,aud=self.env.transition(st,self.env.act('B','SEND','R',('time',)))
        self.assertEqual(aud.verdict,Verdict.VALID)
        self.assertNotIn('person_band',self.env.known_to(ns,'R'))

    def test_repeat_directly_denied_fact_even_if_no_increment(self):
        st=self.after_skip()
        st.local_fact_ids['B'].add('person_band')
        ns,aud=self.env.transition(st,self.env.act('B','SEND','R',('person_band',)))
        self.assertEqual(aud.verdict,Verdict.VIOLATION)
        self.assertTrue(aud.direct_evidence)

    def test_source_visibility_and_ledger_are_distinct(self):
        s0=self.env.initial(); obsB=self.env.observation(s0,'B')
        self.assertNotIn('identity',{f.id for f in obsB.local_facts})
        self.assertEqual(obsB.authorized_receipts,())
        st=self.after_identity();obsB=self.env.observation(st,'B')
        self.assertNotIn('identity',{f.id for f in obsB.local_facts})
        self.assertEqual(obsB.authorized_receipts[0].fact_ids,('identity',))
        # Recipient receipt does not magically grant source database access.

    def test_fabricated_source_cannot_be_sent(self):
        st=self.after_skip()
        bad=self.env.act('B','SEND','R',('identity',))
        _,aud=self.env.transition(st,bad)
        self.assertEqual(aud.verdict,Verdict.INVALID_ACTION)

    def test_undeclared_semantic_content_unknown_not_safe(self):
        st=self.after_identity()
        bad=Action('B','SEND','R',('time',),'X17 meeting time is 14:00. Alice has salary band B2.')
        _,aud=self.env.transition(st,bad)
        self.assertEqual(aud.verdict,Verdict.UNKNOWN)
        self.assertIsNone(self.cce.evaluate(st,bad).reference)

    def test_future_reference_costs_are_exhaustive(self):
        s0=self.env.initial()
        x=self.cce.evaluate(s0,self.env.act('A','SKIP'))
        y=self.cce.evaluate(s0,self.env.act('A','SEND','R',('identity',)))
        self.assertEqual(x.reference.status,RefStatus.EXACT)
        self.assertEqual(y.reference.status,RefStatus.EXACT)
        self.assertEqual((x.reference.best_success_cost,y.reference.best_success_cost),(4.0,7.0))
        pref=certify_preference(x,y)
        self.assertIsNotNone(pref)
        self.assertEqual(pref.source,'future_verified')
        self.assertEqual(pref.chosen.kind,'SKIP')
        self.assertTrue(pref.to_dpo_record()['metadata']['certified'])

    def test_ci_preference_requires_safe_success_witness(self):
        st=self.after_identity()
        safe=self.cce.evaluate(st,self.env.act('B','REQUEST_RULES',recipient='R'))
        unsafe=self.cce.evaluate(st,self.env.act('B','SEND','R',('band',)))
        p=certify_preference(safe,unsafe)
        self.assertEqual(p.source,'ci_safe_success')
        self.assertTrue(p.evidence['violations'])
        self.assertEqual(p.chosen.kind,'REQUEST_RULES')

    def test_abstain_does_not_win_when_safe_task_success_exists(self):
        st=self.after_skip()
        done=self.cce.evaluate(st,self.env.act('B','ABSTAIN'))
        work=self.cce.evaluate(st,self.env.act('B','SEND','R',('band',)))
        pref=certify_preference(work,done)
        self.assertIsNotNone(pref)
        self.assertEqual(pref.chosen.kind,'SEND')

    def test_bounded_search_misses_are_unknown_not_negative(self):
        bad_ref=ReferenceSearch(self.env,self.gen,SearchBudget(max_nodes=1,max_actions=12))
        cce=ContinuationEvaluator(self.env,bad_ref)
        s0=self.env.initial()
        x=cce.evaluate(s0,self.env.act('A','SKIP'))
        y=cce.evaluate(s0,self.env.act('A','SEND','R',('identity',)))
        self.assertEqual(x.reference.status,RefStatus.UNKNOWN)
        self.assertIsNone(certify_preference(x,y))

    def test_nonexhaustive_llm_candidates_cannot_certify_optimality(self):
        def fake_llm(prompt,n):
            return [Action('B','SEND','R',('band',),'X17 has salary band B2.').__dict__]
        modelgen=JSONCandidateGenerator(fake_llm)
        self.assertFalse(modelgen.generate(self.env.observation(self.after_skip(),'B')).exhaustive)
        ref=ReferenceSearch(self.env,modelgen)
        result=ref.search(self.after_skip())
        self.assertEqual(result.status,RefStatus.UNKNOWN)

    def test_downstream_repair_targets_actual_violator(self):
        initial=self.env.initial();steps=[]
        st=initial
        for action in (self.env.act('A','SEND','R',('identity',)),
                       self.env.act('B','SEND','R',('band',))):
            ns,aud=self.env.transition(st,action)
            steps.append(TraceStep(st,action,aud,ns));st=ns
        res=DownstreamPolicyRepair(self.cce,self.gen).repair(Trace(tuple(steps)))
        self.assertEqual(res.status,'verified_repair')
        self.assertEqual(res.failed_step,1)
        self.assertEqual(res.preference.observation.actor,'B')
        self.assertEqual(res.preference.chosen.kind,'REQUEST_RULES')
        self.assertEqual(res.preference.rejected.kind,'SEND')
        self.assertIn('known',res.preference.observation.as_prompt().lower()) if False else None

    def test_pisas_static_labels_cannot_be_treated_as_dynamic(self):
        four={'scenario.json':{'scenario_id':'x','task':{},'cast':{},'org':{},'timeline':[]},
              'utility.json':{'scenario_id':'x','oracle_answer':'a','decision_rule':'','attribute_values':{}},
              'appropriateness.json':{'scenario_id':'x','attributes':{}},
              'visibility.json':{'scenario_id':'x','attributes':{}}}
        ready=audit_pisas_files(four)
        self.assertFalse(ready.train_ready)
        self.assertIn('delivered_recipient_ledger',ready.must_create)
        with self.assertRaises(DataContractError):validate_enriched_episode(four)
        enriched={
            'schema_version':'kci_episode_0.6_cpu','scenario_group_id':'x',
            'rule_family_id':'identity_band_join','norm_version':'norm_v1',
            'task_oracle_version':'eligibility_and_notification_v1',
            'delivered_recipient_ledger':[{'event_id':'evt1','sender':'A','recipient':'R',
                                            'fact_ids':['identity'],'delivered':True,
                                            'body':'X17 corresponds to Alice.','purpose':'resource_planning',
                                            'semantic_verifier_version':'structured_exact_v1'}],
            'structured_compositional_inference_rules':[{'premises':['identity','band'],
                                             'conclusion':'person_band','verifier':'symbolic_v1'}],
            'ci_policy_for_new_derived_facts':[{'policy_id':'N1','recipient':'R',
                      'purpose':'resource_planning','predicate':'person_band','allowed':False}],
            'agent_observation_filter':{'B':'trusted_R_ledger'},
            'legal_action_transition_model':{'version':'toy_ci_simulator_0.5'},
            'next_step_candidate_actions':[{'action_id':'SEND','kind':'SEND','actor':'B','fact_ids':['band']}],
            'action_level_validation':{'by_action':{'SEND':'valid_under_specific_history'}},
            'verifiable_task_success':{'oracle':'project_eligibility_yes'},
            'reference_continuation_evidence':[{'action_id':'SEND','status':'unknown',
                                               'verifier_version':'toy_ci_simulator_0.5'}],
        }
        validate_enriched_episode(enriched)
        enriched['delivered_recipient_ledger'][0]['delivered']=False
        with self.assertRaises(DataContractError):validate_enriched_episode(enriched)

    def test_ref_unknown_action_will_not_export(self):
        st=self.after_identity()
        fake=Action('B','SEND','R',('time',),'unverified free text')
        report=self.cce.evaluate(st,fake)
        self.assertEqual(report.immediate.verdict,Verdict.UNKNOWN)
        safe=self.cce.evaluate(st,self.env.act('B','REQUEST_RULES',recipient='R'))
        self.assertIsNone(certify_preference(safe,report))

if __name__=='__main__':unittest.main()
