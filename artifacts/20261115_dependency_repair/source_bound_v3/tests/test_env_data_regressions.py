import json
import unittest
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from kci import ToyCIEnv,ToyCandidateGenerator,ReferenceSearch,ContinuationEvaluator,certify_preference
from kci.contracts import Action,Message,Verdict
from kci.datasets import DataContractError,validate_enriched_episode,deterministic_group_split,export_verified_preferences
from kci.toy_env import FACTS


class EnvironmentRegressionTests(unittest.TestCase):
    def test_initial_background_and_visible_delivered_ledger_enter_knowledge(self):
        env=ToyCIEnv();state=env.initial()
        self.assertEqual(env.known_to(state,'B'),{'band','time'})
        state.local_fact_ids['R'].add('identity')
        state,_=env.transition(state,env.act('A','SKIP'))
        _,audit=env.transition(state,env.act('B','SEND','R',('band',)))
        self.assertEqual(audit.verdict,Verdict.VIOLATION)
        state=env.initial()
        state.receipts.append(Message('pending','A','R',env.purpose,('identity',),env.rendered(('identity',)),False))
        self.assertNotIn('identity',env.known_to(state,'B'))
        state.receipts[0]=replace(state.receipts[0],delivered=True)
        self.assertIn('person_band',env.known_to(state,'B'))

    def test_task_norms_and_costs_match_observable_configuration(self):
        env=ToyCIEnv(purpose='authorized_audit',forbidden_inferences=set(),action_costs={'REQUEST_RULES':3.0})
        obs=env.observation(env.initial(),'R')
        self.assertNotIn('Alice',obs.task)
        self.assertIn('may learn person-to-salary-band',obs.as_prompt())
        self.assertIn('authorized_audit',obs.as_prompt())
        self.assertIn('3.0',obs.as_prompt())
        state,_=env.transition(env.initial(),env.act('A','SEND','R',('identity',)))
        _,audit=env.transition(state,env.act('B','SEND','R',('band',)))
        self.assertEqual(audit.verdict,Verdict.VALID)

    def test_unknown_recipient_is_invalid_and_returns_an_independent_state(self):
        env=ToyCIEnv();state,_=env.transition(env.initial(),env.act('A','SKIP'))
        after,audit=env.transition(state,env.act('B','SEND','missing',('band',)))
        self.assertEqual(audit.verdict,Verdict.INVALID_ACTION)
        self.assertIsNot(after,state)
        after.local_fact_ids['B'].clear()
        self.assertIn('band',state.local_fact_ids['B'])

    def test_non_message_fields_and_decision_body_cannot_hide_text(self):
        env=ToyCIEnv();state=env.initial()
        _,audit=env.transition(state,Action('A','SKIP',body='Alice has salary band B2.'))
        self.assertEqual(audit.verdict,Verdict.INVALID_ACTION)
        state,_=env.transition(state,env.act('A','SKIP'))
        state,_=env.transition(state,env.act('B','SEND','R',('band',)))
        _,audit=env.transition(state,Action('R','FINALIZE',answer='yes',body='Alice has salary band B2.'))
        self.assertEqual(audit.verdict,Verdict.UNKNOWN)

    def test_decision_messages_do_not_encode_hidden_correctness(self):
        env=ToyCIEnv();state,_=env.transition(env.initial(),env.act('A','SKIP'))
        state,_=env.transition(state,env.act('B','SEND','R',('time',)))
        supported=deepcopy(state);supported.local_fact_ids['R'].add('band')
        unsupported,audit=env.transition(state,env.act('R','FINALIZE',answer='yes'))
        correct,_=env.transition(supported,env.act('R','FINALIZE',answer='yes'))
        self.assertEqual(audit.verdict,Verdict.VALID)
        self.assertFalse(unsupported.final_correct)
        self.assertTrue(correct.final_correct)
        self.assertEqual(unsupported.receipts[-1],correct.receipts[-1])
        self.assertEqual(unsupported.receipts[-1].fact_ids,())
        self.assertIn(unsupported.receipts[-1],env.observation(unsupported,'A').authorized_receipts)
        notified,_=env.transition(correct,env.act('A','NOTIFY','Alice',answer='yes'))
        self.assertTrue(notified.success)
        self.assertEqual(notified.receipts[-1].recipient,'Alice')
        self.assertEqual(notified.receipts[-1].body,env.decision_body('yes'))

    def test_renamed_facts_preserve_join_and_notification_target(self):
        def rename(s):return s.replace('Alice','Mei').replace('X17','Q91').replace('X18','Q92').replace('B2','B3')
        facts={k:replace(f,subject=rename(f.subject),value=rename(f.value),text=rename(f.text)) for k,f in FACTS.items()}
        env=ToyCIEnv(facts=facts,world_id='renamed')
        state,_=env.transition(env.initial(),env.act('A','SEND','R',('identity',)))
        _,audit=env.transition(state,env.act('B','SEND','R',('band',)))
        self.assertEqual(audit.verdict,Verdict.VIOLATION)
        state=env.initial();state.actor='A';state.finalized=True
        batch=ToyCandidateGenerator().generate(env.observation(state,'A'))
        self.assertEqual({a.recipient for a in batch.actions if a.kind=='NOTIFY'},{'Mei'})

    def test_declared_safe_bundles_are_enumerated(self):
        env=ToyCIEnv();gen=ToyCandidateGenerator()
        first=gen.generate(env.observation(env.initial(),'A'))
        self.assertTrue(first.exhaustive)
        self.assertTrue(any(set(a.fact_ids)=={'identity','identity_other'} for a in first.actions))
        self.assertTrue(any(a.kind=='ABSTAIN' for a in first.actions))
        state,_=env.transition(env.initial(),env.act('A','SKIP'))
        batch=gen.generate(env.observation(state,'B'))
        self.assertTrue(any(set(a.fact_ids)=={'band','time'} for a in batch.actions))

    def test_semantic_configuration_versions_are_distinct(self):
        default=ToyCIEnv();changed=ToyCIEnv(action_costs={'SEND':2})
        self.assertNotEqual(default.version,changed.version)
        self.assertNotEqual(default.initial().signature(),changed.initial().signature())
        version=default.version
        default.forbidden_inferences.clear()
        self.assertNotEqual(default.version,version)


class DataRegressionTests(unittest.TestCase):
    def example(self):
        return json.loads((Path(__file__).resolve().parents[1]/'data'/'example_episode_contract.json').read_text(encoding='utf-8'))

    def test_example_contract_and_event_failures(self):
        example=self.example();validate_enriched_episode(example)
        for field in ('event_id','body','purpose','semantic_verifier_version'):
            broken=deepcopy(example);broken['delivered_recipient_ledger'][0].pop(field)
            with self.assertRaises(DataContractError):validate_enriched_episode(broken)
        broken=deepcopy(example);broken['delivered_recipient_ledger']*=2
        with self.assertRaises(DataContractError):validate_enriched_episode(broken)

    def test_reference_is_bound_to_candidate_and_bounded_evidence(self):
        example=self.example()
        for field in ('horizon','search_budget','generator_version','witness_actions'):
            broken=deepcopy(example);broken['reference_continuation_evidence'][0].pop(field)
            with self.assertRaises(DataContractError):validate_enriched_episode(broken)
        broken=deepcopy(example);broken['reference_continuation_evidence'][0]['action_id']='missing'
        with self.assertRaises(DataContractError):validate_enriched_episode(broken)

    def test_statistical_evidence_is_explicit_and_structurally_supported(self):
        example=self.example()
        example['reference_continuation_evidence']=[{
            'action_id':'a1','status':'estimated_preference','verifier_version':'v1',
            'samples':2,'delta':.05,'sample_seeds':[1,2],'reference_policy_version':'frozen_v1',
            'epistemic_status':'statistical_estimate_NOT_optimality_proof',
            'comparison_metric':'ci_compliance','confidence_interval':[.1,.4]}]
        validate_enriched_episode(example)
        example['reference_continuation_evidence'][0]['sample_seeds']=[1,1]
        with self.assertRaises(DataContractError):validate_enriched_episode(example)

    def test_rule_family_holdout_keeps_the_whole_world_together(self):
        records=[{'metadata':{'scenario_group_id':world,'rule_family_id':family}}
                 for world,family in [('a','holdout'),('a','other'),('b','holdout'),('c','other')]]
        splits=deterministic_group_split(records,held_out_rule_families={'holdout':'test'})
        self.assertEqual(len([r for r in splits['test'] if r['metadata']['scenario_group_id'] in {'a','b'}]),3)
        with self.assertRaises(DataContractError):
            deterministic_group_split(records,held_out_rule_families={'holdout':'test','other':'validation'})

    def test_export_hashes_bind_prompt_actions_and_evidence(self):
        env=ToyCIEnv();cce=ContinuationEvaluator(env,ReferenceSearch(env,ToyCandidateGenerator()))
        state=env.initial()
        pref=certify_preference(cce.evaluate(state,env.act('A','SKIP')),
                               cce.evaluate(state,env.act('A','SEND','R',('identity',))))
        rows=export_verified_preferences([pref,pref])
        self.assertEqual(len(rows),1)
        row=rows[0];digest=row['metadata'].pop('record_sha256')
        canonical=json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        self.assertEqual(digest,sha256(canonical.encode('utf-8')).hexdigest())


if __name__=='__main__':unittest.main()
