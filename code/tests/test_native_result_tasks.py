import unittest
from copy import deepcopy
from unittest.mock import patch
from experiments.receipt_dataset import make_world,SOURCE
from kci.native_result_tasks import NativeResultTask,result_edges


class NativeResultTests(unittest.TestCase):
    def ready(self):
        env=NativeResultTask(make_world(47771,'query'),SOURCE);s=env.initial()
        result=next(m['content']['result'] for m in s.ledger if m['content'].get('type')=='computed_native_result')
        edges=result_edges('query',result);selected=[edges[0],next(e for e in edges if e[1]!=edges[0][1])]
        env.world['providers']=[{'ticket':e[0],'links':[e]} for e in selected]
        env.world['histories']=[[selected[0][1],'secret']]
        return env,env.initial()

    def test_native_output_creates_future_value_difference(self):
        env,s=self.ready();p=env.preference(s,*env.actions(s))
        self.assertEqual(p['immediate'],['VALID','VALID']);self.assertEqual(sorted(p['values']),[(1,0,0),(1,1,0)])
        env.world['histories'][0][0]=env.world['providers'][1]['links'][0][1]
        other=env.preference(env.initial(),*env.actions(env.initial()));self.assertEqual(other['chosen'],1-p['chosen'])

    def test_no_additional_receipt_or_binding_field_is_published(self):
        env,s=self.ready();p=env.preference(s,*env.actions(s));s,_=env.transition(s,env.actions(s)[p['chosen']])
        final,audit=env.transition(s,'PUBLISH_FULL')
        self.assertEqual(final.ledger[-1]['content'],{'type':'native_result','result':env.expected})
        self.assertTrue(final.success);self.assertEqual(final.diagnostic_steps,2)

    def test_publication_consumes_actual_result_without_source_reread(self):
        env,s=self.ready();s,_=env.transition(s,env.actions(s)[1])
        with patch.object(env.kernel,'source_value',side_effect=AssertionError('private reread')):
            final,audit=env.transition(s,'PUBLISH_FULL')
        self.assertEqual(audit['verdict'],'VALID');self.assertTrue(final.success)

    def test_expected_answer_is_not_a_policy_observation(self):
        env,s=self.ready();before=env.observation(s);env.expected=[]
        self.assertEqual(before,env.observation(s))
        s,_=env.transition(s,env.actions(s)[1]);final,audit=env.transition(s,'PUBLISH_FULL')
        self.assertFalse(final.success);self.assertNotEqual(final.ledger[-1]['content']['result'],[])


if __name__=='__main__':unittest.main()
