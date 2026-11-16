import concurrent.futures
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from kci import (ToyCIEnv,ToyCandidateGenerator,JSONCandidateGenerator,ReferenceSearch,
                 ContinuationEvaluator,DeterministicPolicy,certify_preference,
                 export_verified_preferences,validate_enriched_episode)
from kci.contracts import Trace
from kci.datasets import DataContractError
from kci.rewards import outcome_from_trace


class BatchIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.env=ToyCIEnv();self.gen=ToyCandidateGenerator()
        self.reference=ReferenceSearch(self.env,self.gen)
        self.cce=ContinuationEvaluator(self.env,self.reference)

    def before_notification(self):
        state=self.env.initial()
        for action in (self.env.act('A','SKIP'),self.env.act('B','SEND','R',('band',)),
                       self.env.act('R','FINALIZE',answer='yes')):
            state,_=self.env.transition(state,action)
        return state

    def test_returned_reference_cannot_poison_cached_witness(self):
        state=self.env.initial();action=self.env.act('A','SKIP')
        first=self.cce.evaluate(state,action)
        expected=first.reference.best_trace.steps[0].after.cost
        first.reference.best_trace.steps[0].after.cost=999
        second=self.cce.evaluate(state,action)
        self.assertEqual(second.reference.best_trace.steps[0].after.cost,expected)
        second.reference.best_trace.steps[0].after.cost=888
        third=self.cce.evaluate(state,action)
        self.assertEqual(third.reference.best_trace.steps[0].after.cost,expected)

    def test_reference_and_actual_trace_do_not_alias_caller_state(self):
        state,_=self.env.transition(self.env.initial(),self.env.act('A','SKIP'))
        result=self.reference.search(state)
        result.best_trace.steps[0].before.cost=999
        self.assertEqual(state.cost,1)
        policy=DeterministicPolicy(lambda obs:self.env.act(obs.actor,'ABSTAIN'),'frozen_abort')
        trace=ContinuationEvaluator(self.env,None,policy).rollout_actual(state)
        trace.steps[0].before.cost=888
        self.assertEqual(state.cost,1)

    def test_parallel_same_state_evaluations_are_deterministic_and_isolated(self):
        state=self.env.initial();snapshot=state.signature()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            reports=list(pool.map(lambda i:self.cce.evaluate(state,self.env.act('A','SKIP') if i%2==0
                              else self.env.act('A','SEND','R',('identity',))),range(16)))
        self.assertEqual([r.reference.best_success_cost for r in reports],[4.,7.]*8)
        self.assertEqual(state.signature(),snapshot)
        reports[0].reference.best_trace.steps[0].after.cost=999
        self.assertNotEqual(reports[2].reference.best_trace.steps[0].after.cost,999)

    def test_non_object_json_candidates_are_rejected_without_crashing(self):
        inputs=['["actor","kind"]','[]','null','42','{broken',{'actor':'A','kind':'SKIP'}]
        batch=JSONCandidateGenerator(lambda prompt,n:inputs).generate(self.env.observation(self.env.initial(),'A'))
        self.assertEqual(batch.rejected_parse_count,5)
        self.assertEqual(len(batch.actions),1);self.assertEqual(batch.actions[0].kind,'SKIP')

    def test_terminal_candidate_has_known_outcome_and_zero_step_witness(self):
        state=self.before_notification()
        def never_called(obs):raise AssertionError('terminal suffix cannot call policy')
        cce=ContinuationEvaluator(self.env,self.reference,DeterministicPolicy(never_called,'terminal_frozen'))
        report=cce.evaluate(state,self.env.act('A','NOTIFY','Alice',answer='yes'),actual_rollouts=2)
        self.assertEqual(report.reference.best_trace.steps,())
        self.assertTrue(report.reference.best_trace.final.success)
        for trace in report.actual_traces:
            outcome=outcome_from_trace(trace)
            self.assertIs(outcome.safe_success,True);self.assertEqual(outcome.execution_cost,4)

    def test_empty_trace_requires_observed_terminal_state_and_copies_it(self):
        self.assertIsNone(outcome_from_trace(Trace(())).safe_success)
        state=self.env.initial()
        self.assertIsNone(outcome_from_trace(Trace((),start_state=state)).safe_success)
        state.done=True;state.success=True;state.cost=4
        trace=Trace((),start_state=state)
        state.success=False
        self.assertIs(outcome_from_trace(trace).safe_success,True)
        trace.final.cost=999
        self.assertEqual(state.cost,4)

    def terminal_episode(self):
        state=self.before_notification()
        report=self.cce.evaluate(state,self.env.act('A','NOTIFY','Alice',answer='yes'))
        final=report.reference.best_trace.final
        episode=json.loads((Path(__file__).resolve().parents[1]/'data'/'example_episode_contract.json').read_text(encoding='utf-8'))
        episode['norm_version']=final.norm_version;episode['task_oracle_version']=final.oracle_version
        item=episode['reference_continuation_evidence'][0]
        item['witness_actions']=[];item['safe_success_cost']=final.cost
        item['after_state_id']=self.cce.state_id(final)
        item['terminal_state']={'state_id':item['after_state_id'],'done':final.done,'success':final.success,
                                'any_violation':final.any_violation,'cost':final.cost,
                                'norm_version':final.norm_version,'oracle_version':final.oracle_version}
        return episode

    def test_empty_success_suffix_requires_matching_terminal_evidence(self):
        episode=self.terminal_episode();validate_enriched_episode(episode)
        for field in ('terminal_state','after_state_id'):
            broken=deepcopy(episode);broken['reference_continuation_evidence'][0].pop(field)
            with self.assertRaises(DataContractError):validate_enriched_episode(broken)
        for field,value in (('done',False),('success',False),('any_violation',True),
                            ('state_id','different'),('norm_version','different'),
                            ('oracle_version','different'),('cost',99)):
            broken=deepcopy(episode);broken['reference_continuation_evidence'][0]['terminal_state'][field]=value
            with self.assertRaises(DataContractError):validate_enriched_episode(broken)

    def test_hidden_recipient_background_reverse_labels_are_blocked_and_preserved(self):
        normal,_=self.env.transition(self.env.initial(),self.env.act('A','SKIP'))
        hidden=deepcopy(normal);hidden.local_fact_ids['R'].add('identity')
        self.assertEqual(self.env.observation(normal,'B'),self.env.observation(hidden,'B'))
        band=self.env.act('B','SEND','R',('band',));request=self.env.act('B','REQUEST_RULES','R')
        preferences=[certify_preference(self.cce.evaluate(s,band),self.cce.evaluate(s,request))
                     for s in (normal,hidden)]
        before=[p.to_dpo_record() for p in preferences]
        with self.assertRaises(DataContractError) as caught:export_verified_preferences(preferences)
        records=caught.exception.conflicting_records
        self.assertEqual(len(records),2)
        self.assertEqual(records[0]['chosen'],records[1]['rejected'])
        self.assertEqual(records[0]['prompt'],records[1]['prompt'])
        self.assertEqual(before,[p.to_dpo_record() for p in preferences])

    def test_export_allows_same_direction_and_different_visible_prompts(self):
        state=self.env.initial()
        pref=certify_preference(self.cce.evaluate(state,self.env.act('A','SKIP')),
                               self.cce.evaluate(state,self.env.act('A','SEND','R',('identity',))))
        same_direction=replace(pref,state_id='another_state')
        self.assertEqual(len(export_verified_preferences([pref,same_direction])),2)
        visible_other=replace(pref,chosen=pref.rejected,rejected=pref.chosen,state_id='visible_other',
                              observation=replace(pref.observation,task=pref.observation.task+' Other context.'))
        self.assertEqual(len(export_verified_preferences([pref,visible_other])),2)


if __name__=='__main__':unittest.main()
