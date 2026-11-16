import unittest
from copy import deepcopy
from unittest.mock import patch
from experiments.receipt_dataset import make_world,rename,SOURCE
from kci.receipt_tasks import ReceiptTask,closure


class ReceiptTaskTests(unittest.TestCase):
    def env(self,kind='meeting',**kw):return ReceiptTask(make_world(911,kind,**kw),SOURCE)

    def publish_state(self,env):
        s=env.initial();choices=env.actions(s)
        proof=env.preference(s,*choices);action=choices[proof['chosen'] if proof['chosen'] is not None else 0]
        for a in (action,'DELIVER_INPUTS'):s,audit=env.transition(s,a);self.assertEqual(audit['verdict'],'VALID')
        return s

    def test_both_first_actions_legal_but_success_values_differ_without_cost(self):
        for kind in ('meeting','query'):
            env=self.env(kind);s=env.initial();p=env.preference(s,*env.actions(s))
            self.assertEqual(p['immediate'],['VALID','VALID']);self.assertIsNotNone(p['chosen'])
            self.assertEqual(sorted(p['values']),[(1,0,0),(1,1,0)])

    def test_removing_obligation_removes_long_term_preference(self):
        for kw in ({'required':False},{'allow':True},{'delivered':False},{'mode':'none'}):
            env=self.env(**kw);s=env.initial();p=env.preference(s,*env.actions(s))
            self.assertIsNone(p['chosen']);self.assertEqual(p['values'],[(1,1,0),(1,1,0)])

    def test_no_safe_success_is_not_exported_as_preference(self):
        env=self.env(mode='both');s=env.initial();p=env.preference(s,*env.actions(s))
        self.assertEqual(p['values'],[(1,0,0),(1,0,0)]);self.assertIsNone(p['chosen'])

    def test_wait_and_reroute_never_change_values(self):
        env=self.env();s=env.initial();expected=env.preference(s,*env.actions(s))['values']
        for _ in range(11):
            for a in ('WAIT','REROUTE_COMPUTATION'):s,_=env.transition(s,a)
        self.assertEqual(env.preference(s,*env.actions(s))['values'],expected)

    def test_symbol_renaming_and_candidate_order(self):
        for seed in range(4):
            world=make_world(seed);env=ReceiptTask(world,SOURCE);s=env.initial();p=env.preference(s,*env.actions(s))
            changed=rename(world,seed+400);changed['providers'].reverse();other=ReceiptTask(changed,SOURCE);t=other.initial()
            self.assertEqual(other.preference(t,*other.actions(t))['chosen'],1-p['chosen'])

    def test_all_required_messages_are_consumed(self):
        for kind in ('meeting','query'):
            env=self.env(kind);s=self.publish_state(env)
            for i,m in enumerate(s.ledger):
                if m['recipient']!='Worker':continue
                bad=deepcopy(s);bad.ledger.pop(i);after,audit=env.transition(bad,'PUBLISH_FULL')
                self.assertEqual(audit['verdict'],'UNKNOWN');self.assertEqual(after,bad)

    def test_runtime_cannot_reread_private_sources(self):
        for kind in ('meeting','query'):
            env=self.env(kind);s=self.publish_state(env)
            with patch.object(env.kernel,'source_value',side_effect=AssertionError('hidden reread')):
                with patch.object(env.kernel,'business_constraints',side_effect=AssertionError('hidden reread')):
                    after,audit=env.transition(s,'PUBLISH_FULL')
            self.assertEqual(audit['verdict'],'VALID');self.assertTrue(after.success)

    def test_changed_delivered_value_affects_computation(self):
        env=self.env('query');s=self.publish_state(env)
        m=next(m for m in s.ledger if m['content'].get('type')=='company_gas_stations')
        m['content']['value'][0]['gas_stations_count']+=91;m['sha256']=env.kernel.evidence_hash(m)
        after,audit=env.transition(s,'PUBLISH_FULL');self.assertEqual(audit['verdict'],'VALID');self.assertFalse(after.success)
        result=after.ledger[-1]['content']['result'];self.assertEqual(result[0]['stations'],env.expected[0]['stations']+91)

    def test_source_permission_revocation(self):
        env=self.env();s,_=env.transition(env.initial(),env.actions(env.initial())[0]);env.kernel.source_acl['business_constraints']=set()
        after,audit=env.transition(s,'DELIVER_INPUTS');self.assertEqual(audit['verdict'],'UNKNOWN');self.assertEqual(after,s)

    def test_task_redaction_failure_is_obligation_sensitive(self):
        for required in (False,True):
            env=self.env(required=required);s=self.publish_state(env);after,_=env.transition(s,'PUBLISH_MINIMAL')
            self.assertEqual(after.success,not required)

    def test_set_closure_handles_branching_and_order(self):
        bodies=[{'identities':[['p','a']]},{'links':[['a','b'],['a','c'],['b','d']]},
                {'histories':[['c','x'],['d','y']]}]
        ledger=[{'recipient':'R','content':b} for b in bodies]
        self.assertEqual(closure(ledger),{('p','x'),('p','y')});self.assertEqual(closure(ledger[::-1]),closure(ledger))

    def test_recipient_knowing_bindings_changes_future_to_immediate_violation(self):
        env=self.env();s=env.initial();actions=env.actions(s);proof=env.preference(s,*actions)
        unsafe=actions[1-proof['chosen']]
        env.kernel.send(s,'Controller','R',{'links':[edge for p in env.world['providers'] for edge in p['links']]})
        _,audit=env.transition(s,unsafe)
        self.assertEqual(audit['verdict'],'VIOLATION')


if __name__=='__main__':unittest.main()
