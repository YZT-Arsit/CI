import json
import unittest
from dataclasses import replace

from kci.contracts import Action,Trace,TraceStep,Verdict,RefStatus
from kci.continuation import ContinuationEvaluator,ReferenceSearch,certify_preference
from kci.repair import DownstreamPolicyRepair
from kci.multi_env import MultiRuleCIEnv,FAMILIES,build_case,canonical,canonical_source_text
from kci.multi_candidates import MultiRuleCandidateGenerator


def replace_source(env,key,*,value=None,subject=None,predicate=None):
    facts={k:f for k,f in env.facts.items() if k.startswith(('context_','payload_')) or k=='notification_target'}
    fact=facts[key]
    text_value=canonical(value) if value is not None else fact.value
    changed=replace(fact,value=text_value,subject=subject or fact.subject,predicate=predicate or fact.predicate)
    facts[key]=replace(changed,text=canonical_source_text(changed))
    return MultiRuleCIEnv(env.family,facts,json.loads(env.facts['task_rule'].value),world_id='controlled_variant')


class MultiEnvironmentTests(unittest.TestCase):
    def test_source_text_must_bind_all_structured_semantics(self):
        for family in FAMILIES:
            env=build_case(family)
            sources={k:f for k,f in env.facts.items() if k.startswith(('context_','payload_')) or k=='notification_target'}
            for key in sources:
                for change in ({'text':sources[key].text+' altered'}, {'subject':'DifferentSubject'}):
                    bad=dict(sources);bad[key]=replace(bad[key],**change)
                    with self.subTest(family=family,key=key,change=change):
                        with self.assertRaises(ValueError):
                            MultiRuleCIEnv(family,bad,json.loads(env.facts['task_rule'].value),world_id='bad_text')
            bad=dict(sources);bad['notification_target']=replace(bad['notification_target'],value='DifferentPerson')
            with self.assertRaises(ValueError):
                MultiRuleCIEnv(family,bad,json.loads(env.facts['task_rule'].value),world_id='bad_target')
            bad=dict(sources);payload=bad['payload_0']
            value=json.loads(payload.value)
            if isinstance(value,int):value=1
            elif family=='aggregate_difference':value=dict(value,total=1)
            else:value=dict(value,members=['DifferentOne','DifferentTwo'])
            bad['payload_0']=replace(payload,value=canonical(value))
            with self.assertRaises(ValueError):
                MultiRuleCIEnv(family,bad,json.loads(env.facts['task_rule'].value),world_id='bad_value')

    def cce(self,env):return ContinuationEvaluator(env,ReferenceSearch(env,MultiRuleCandidateGenerator()))

    def after_context(self,env,ids=None):
        state,audit=env.transition(env.initial(),env.act('A','SEND','R',env.context_ids if ids is None else ids))
        self.assertEqual(audit.verdict,Verdict.VALID)
        return state

    def fast_path(self,env,answer):
        state=env.initial()
        for action in (env.act('A','SKIP'),env.act('B','SEND','R',('payload_0',)),
                       env.act('R','FINALIZE',answer=answer),env.act('A','NOTIFY',env.target,answer=answer)):
            state,audit=env.transition(state,action);self.assertEqual(audit.verdict,Verdict.VALID)
        return state

    def test_balanced_answers_operations_and_exact_reference_paths(self):
        for family in FAMILIES:
            for op in (('membership',) if family=='set_intersection' else ('threshold','parity')):
                for answer in ('yes','no'):
                    with self.subTest(family=family,op=op,answer=answer):
                        env=build_case(family,answer,task_op=op);cce=self.cce(env);state=env.initial()
                        self.assertEqual(env.expected_answer,answer)
                        skip=cce.evaluate(state,env.act('A','SKIP'))
                        context=cce.evaluate(state,env.act('A','SEND','R',env.context_ids))
                        self.assertEqual((skip.reference.status,context.reference.status),(RefStatus.EXACT,RefStatus.EXACT))
                        self.assertEqual((skip.reference.best_success_cost,context.reference.best_success_cost),(4,7))
                        self.assertEqual(certify_preference(skip,context).chosen.kind,'SKIP')
                        self.assertTrue(self.fast_path(env,answer).success)
                        self.assertFalse(self.fast_path(env,'no' if answer=='yes' else 'yes').success)

    def test_task_answer_changes_with_actual_numbers(self):
        env=build_case('identity_attribute','yes')
        changed=replace_source(env,'payload_0',value=1)
        self.assertEqual(changed.expected_answer,'no')
        self.assertTrue(self.fast_path(changed,'no').success)
        self.assertFalse(self.fast_path(changed,'yes').success)

    def test_same_payload_history_flip_and_dpr_for_all_grammars(self):
        for family in FAMILIES:
            env=build_case(family,'no');state=env.initial()
            context=env.act('A','SEND','R',env.context_ids);after,first=env.transition(state,context)
            payload=env.act('B','SEND','R',('payload_0',));leaked,audit=env.transition(after,payload)
            self.assertEqual(audit.verdict,Verdict.VIOLATION)
            self.assertTrue(audit.inference_evidence)
            skip,_=env.transition(state,env.act('A','SKIP'))
            _,safe=env.transition(skip,payload);self.assertEqual(safe.verdict,Verdict.VALID)
            trace=Trace((TraceStep(state,context,first,after),TraceStep(after,payload,audit,leaked)))
            repair=DownstreamPolicyRepair(self.cce(env),MultiRuleCandidateGenerator()).repair(trace)
            self.assertEqual(repair.status,'verified_repair');self.assertEqual(repair.failed_step,1)
            self.assertEqual(repair.preference.chosen.kind,'REQUEST_RULES')

    def test_alias_chain_needs_all_three_premises_and_allows_partial_join(self):
        env=build_case('alias_chain_3')
        for ids in (('context_0',),('context_1',)):
            state=self.after_context(env,ids)
            _,audit=env.transition(state,env.act('B','SEND','R',('payload_0',)))
            self.assertEqual(audit.verdict,Verdict.VALID)
        full=self.after_context(env)
        self.assertTrue(any(x.startswith('person_case:') for x in env.known_to(full,'R')))
        self.assertFalse(any(x.startswith('protected:') for x in env.known_to(full,'R')))
        _,audit=env.transition(full,env.act('B','SEND','R',('payload_0',)))
        self.assertIn('context_0,context_1,payload_0',audit.inference_evidence[0])
        mismatched=replace_source(env,'context_1',subject='DifferentEmployee')
        state=self.after_context(mismatched)
        _,audit=mismatched.transition(state,mismatched.act('B','SEND','R',('payload_0',)))
        self.assertEqual(audit.verdict,Verdict.VALID)

    def test_aggregate_scopes_and_single_missing_member_are_required(self):
        env=build_case('aggregate_difference');base=json.loads(env.facts['context_0'].value)
        for field,value in (('period','another_period'),('unit','other_units'),
                            ('population',['One','Two','Three']),('contributors',base['population'])):
            altered=dict(base);altered[field]=value;other=replace_source(env,'context_0',value=altered)
            state=self.after_context(other)
            _,audit=other.transition(state,other.act('B','SEND','R',('payload_0',)))
            self.assertEqual(audit.verdict,Verdict.VALID)
        permuted=dict(base);permuted['population']=list(reversed(base['population']))
        other=replace_source(env,'context_0',value=permuted);state=self.after_context(other)
        _,audit=other.transition(state,other.act('B','SEND','R',('payload_0',)))
        self.assertEqual(audit.verdict,Verdict.VIOLATION)
        fractional=dict(base);fractional['total']=1.5
        with self.assertRaises(ValueError):replace_source(env,'context_0',value=fractional)

    def test_set_intersection_empty_multiple_and_scope_mismatch_do_not_identify(self):
        env=build_case('set_intersection');base=json.loads(env.facts['context_0'].value)
        payload=json.loads(env.facts['payload_0'].value)
        controls=[dict(base,members=['AbsentOne','AbsentTwo']),dict(base,members=payload['members']),
                  dict(base,scope='another_complete_registry')]
        for value in controls:
            other=replace_source(env,'context_0',value=value);state=self.after_context(other)
            _,audit=other.transition(state,other.act('B','SEND','R',('payload_0',)))
            self.assertEqual(audit.verdict,Verdict.VALID)
        for value in (dict(base,complete=False),dict(base,members=[base['members'][0]])):
            with self.assertRaises(ValueError):replace_source(env,'context_0',value=value)

    def test_authorization_flip_preserves_family_group_and_changes_version(self):
        for family in FAMILIES:
            denied=build_case(family);allowed=build_case(family,allow_protected=True)
            self.assertEqual(denied.base_case_group_id,allowed.base_case_group_id)
            self.assertEqual(denied.workflow_group_id,allowed.workflow_group_id)
            self.assertNotEqual(denied.version,allowed.version)
            state=self.after_context(allowed)
            _,audit=allowed.transition(state,allowed.act('B','SEND','R',('payload_0',)))
            self.assertEqual(audit.verdict,Verdict.VALID)

    def test_observation_filter_candidate_input_and_no_initial_answer_oracle(self):
        env=build_case('alias_chain_3','no');state=env.initial()
        obs=env.observation(state,'B')
        self.assertEqual({f.id for f in obs.local_facts},{'payload_0'})
        self.assertNotIn('final_correct',obs.as_prompt());self.assertNotIn('expected_answer',obs.as_prompt())
        self.assertNotIn(env.target,obs.task)
        state=self.after_context(env);obs=env.observation(state,'B')
        self.assertEqual({f.id for f in obs.local_facts},{'payload_0'})
        self.assertEqual(len(obs.authorized_receipts),1)
        batch=MultiRuleCandidateGenerator().generate(obs)
        self.assertTrue(batch.exhaustive);self.assertLessEqual(len(batch.actions),12)
        self.assertTrue(any(a.kind=='REQUEST_RULES' for a in batch.actions))

    def test_canonical_messages_and_permissions(self):
        env=build_case('identity_attribute');state=env.initial();snapshot=state.signature()
        _,bad=env.transition(state,Action('A','SEND','R',('context_0',),'unverified prose'))
        self.assertEqual(bad.verdict,Verdict.UNKNOWN);self.assertEqual(state.signature(),snapshot)
        _,wrong_actor=env.transition(state,env.act('B','SEND','R',('payload_0',)))
        self.assertEqual(wrong_actor.verdict,Verdict.INVALID_ACTION)
        after,_=env.transition(state,env.act('A','SKIP'))
        _,fabricated=env.transition(after,env.act('B','SEND','R',('context_0',)))
        self.assertEqual(fabricated.verdict,Verdict.INVALID_ACTION)
        leaked,audit=env.transition(state,env.act('A','SEND','R',('notification_target',)))
        self.assertEqual(audit.verdict,Verdict.VIOLATION);self.assertTrue(leaked.any_violation)

    def test_local_compute_requires_actual_criterion_delivery_and_final_body_hides_correctness(self):
        env=build_case('identity_attribute','no');state,_=env.transition(env.initial(),env.act('A','SKIP'))
        _,audit=env.transition(state,env.act('B','COMPUTE_TASK'))
        self.assertEqual(audit.verdict,Verdict.INVALID_ACTION)
        for action in (env.act('B','REQUEST_RULES','R'),env.act('R','SEND','B',('task_rule',)),env.act('B','COMPUTE_TASK')):
            state,audit=env.transition(state,action);self.assertEqual(audit.verdict,Verdict.VALID)
        self.assertIn('decision_no',state.local_fact_ids['B']);self.assertNotIn('decision_yes',state.local_fact_ids['B'])
        state,_=env.transition(state,env.act('B','SEND','R',('decision_no',)))
        wrong,_=env.transition(state,env.act('R','FINALIZE',answer='yes'))
        self.assertFalse(wrong.final_correct);self.assertEqual(wrong.receipts[-1].fact_ids,())
        self.assertEqual(wrong.receipts[-1].body,env.decision_body('yes'))


if __name__=='__main__':unittest.main()
