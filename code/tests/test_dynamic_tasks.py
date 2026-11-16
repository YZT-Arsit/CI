import unittest
from pathlib import Path
from copy import deepcopy
from kci.dynamic_tasks import DynamicTask, DynamicState, canonical

SOURCE=Path(__file__).resolve().parents[2]/'artifacts/20261113_iteration2/source_anchors'


class DynamicTaskTests(unittest.TestCase):
    def env(self,kind,**kwargs):return DynamicTask(kind,SOURCE,**kwargs)
    def values(self,e):
        s=e.decision_start();result=[]
        for a in e.options(s):
            n,audit=e.transition(s,a);self.assertEqual(audit['verdict'],'VALID')
            result.append(e.exact(n)['min_total_cost'])
        return result
    def test_history_reverses_future_cost(self):
        for kind in ('meeting','query'):
            related=self.values(self.env(kind));unrelated=self.values(self.env(kind,related=False))
            self.assertGreater(related[0],related[1]);self.assertLess(unrelated[0],unrelated[1])
    def test_authorization_removes_flip(self):
        for kind in ('meeting','query'):
            self.assertEqual(self.values(self.env(kind,allow=True)),self.values(self.env(kind,related=False)))
    def test_deadline_has_safe_alternative(self):
        for kind,horizon in [('meeting',6),('query',7)]:
            value=self.values(self.env(kind,horizon=horizon))
            self.assertIsNone(value[0]);self.assertEqual(value[1],horizon)
    def test_actual_delivery_and_drafts(self):
        e=self.env('meeting');s=e.decision_start();before=canonical(s.ledger)
        n,a=e.transition(s,'SEND_ROSTER');self.assertEqual(canonical(s.ledger),before)
        n,a=e.transition(n,'SEND_CASE_LINK');self.assertEqual(a['verdict'],'VIOLATION')
        self.assertEqual(len(e.protected(n.ledger)),1)
    def test_observation_truth_separation(self):
        a=self.env('query');b=self.env('query',related=False)
        sa,sb=a.decision_start(),b.decision_start()
        self.assertNotEqual(a.observation(sa),b.observation(sb))
        self.assertEqual(a.observation(sa,hide_receipts=True),b.observation(sb,hide_receipts=True))
        self.assertNotIn('expected',canonical(a.observation(sa)))
        obs=a.observation(sa);obs['authorized_receipts'].clear()
        self.assertTrue(sa.ledger)
    def test_unknown_action_cannot_advance(self):
        e=self.env('query');s=e.decision_start();n,a=e.transition(s,'invented')
        self.assertEqual(a['verdict'],'UNKNOWN');self.assertEqual(n,s)
    def test_success_witness_replays_and_oracles(self):
        for kind in ('meeting','query'):
            e=self.env(kind);s=e.decision_start();r=e.exact(s)
            for a in r['witness']:s,audit=e.transition(s,a);self.assertEqual(audit['verdict'],'VALID')
            self.assertTrue(s.success);self.assertEqual(s.result,e.expected)
            self.assertEqual(s.ledger[-1]['recipient'],'Client')
    def test_query_shards_real_join_and_copy(self):
        e=self.env('query',related=False);s=e.decision_start()
        for a in ('OPEN_INDEX','FETCH_CITY_BUNDLE','FETCH_COUNTS','JOIN','DELIVER_RESULT'):s,audit=e.transition(s,a)
        self.assertTrue(s.success)
        self.assertEqual(s.result[0],{'company':'PetroMax','stations':15,'city':'Dallas'})
        s.result[0]['stations']=999
        self.assertEqual(e.expected[0]['stations'],15)


if __name__=='__main__':unittest.main()
