import json,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.receipt_dataset import make_world
from kci.ordering_receipt_tasks import OrderingReceiptTask

ROOT=Path(__file__).resolve().parents[2]
SOURCES=ROOT/'artifacts/20261115_semantic_repair/heldout_sources'


class OrderingReceiptTests(unittest.TestCase):
    def ready(self,number=6):
        folder=SOURCES/f'scenario_{number:02d}'
        env=OrderingReceiptTask(make_world(42,'meeting'),folder/'adapter')
        s=env.initial();actions=env.actions(s);p=env.preference(s,*actions)
        s,_=env.transition(s,actions[p['chosen']]);s,_=env.transition(s,'DELIVER_INPUTS')
        return env,s

    def test_all_native_ordering_oracles_match(self):
        for number in range(6,10):
            env,s=self.ready(number);after,audit=env.transition(s,'PUBLISH_FULL')
            oracle=json.loads((SOURCES/f'scenario_{number:02d}'/'utility.json').read_text())['oracle_schedule']
            self.assertEqual(after.ledger[-1]['content']['result'],oracle);self.assertTrue(after.success)

    def test_ordering_projection_does_not_include_private_reason(self):
        env,s=self.ready()
        m=next(m for m in s.ledger if m['content'].get('type')=='business_constraints')
        ordering=next(c for c in m['content']['value'] if c['type']=='meeting_ordering')
        self.assertEqual(set(ordering),{'person','type','before_meeting_id','after_meeting_id'})

    def test_room_uses_delivered_ordering_not_private_source(self):
        env,s=self.ready()
        with patch.object(env.kernel,'business_constraints',side_effect=AssertionError('private reread')):
            after,audit=env.transition(s,'PUBLISH_FULL')
        self.assertEqual(audit['verdict'],'VALID');self.assertTrue(after.success)

    def test_ordering_reversal_changes_actual_result(self):
        env,s=self.ready()
        m=next(m for m in s.ledger if m['content'].get('type')=='business_constraints')
        ordering=next(c for c in m['content']['value'] if c['type']=='meeting_ordering')
        ordering['before_meeting_id'],ordering['after_meeting_id']=ordering['after_meeting_id'],ordering['before_meeting_id']
        m['sha256']=env.kernel.evidence_hash(m)
        after,audit=env.transition(s,'PUBLISH_FULL')
        self.assertEqual(audit['verdict'],'VALID');self.assertFalse(after.success)
        self.assertNotEqual(after.ledger[-1]['content']['result'],env.expected)


if __name__=='__main__':unittest.main()
