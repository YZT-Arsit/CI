import unittest
from copy import deepcopy
from unittest.mock import patch
from experiments.task_evidence_dependency_probe import dependency_cases,computational_controls,acl_controls,prepare


class TaskEvidenceDependencyTests(unittest.TestCase):
    def test_meeting_task_evidence_dependency(self):self.assertTrue(all(r['passed'] for r in dependency_cases('meeting')))
    def test_query_task_evidence_dependency(self):self.assertTrue(all(r['passed'] for r in dependency_cases('query')))
    def test_meeting_values_actually_affect_schedule(self):self.assertTrue(all(r['passed'] for r in computational_controls('meeting')))
    def test_query_values_actually_affect_join(self):self.assertTrue(all(r['passed'] for r in computational_controls('query')))
    def test_acl_revocation(self):
        for kind in ('meeting','query'):self.assertTrue(all(r['passed'] for r in acl_controls(kind)))
    def test_room_solver_cannot_read_private_source(self):
        env,state=prepare('meeting','PRIVATE_BOOKING ACK_PRIVATE SEND_CASE_LINK')
        with patch.object(env,'business_constraints',side_effect=AssertionError('Room touched private source')):
            after,audit=env.transition(state,'RESERVE')
        self.assertEqual(audit['verdict'],'VALID');self.assertEqual(after.result,env.expected)
        self.assertEqual(after.access_log[-1]['resource'],'public_schedule')
        self.assertEqual(after.access_log[-1]['reader'],'Room')
    def test_projection_forward_cannot_reread_source(self):
        env,state=prepare('query','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES')
        with patch.object(env,'source_value',side_effect=AssertionError('SEND_CITIES reread source')):
            after,audit=env.transition(state,'SEND_CITIES')
        self.assertEqual(audit['verdict'],'VALID')
    def test_missing_ancestor_rejected_and_no_partial_side_effect(self):
        env,state=prepare('meeting','PRIVATE_BOOKING ACK_PRIVATE SEND_CASE_LINK')
        state.ledger=[m for m in state.ledger if m['content']['type']!='private_booking']
        saved=deepcopy(state);after,audit=env.transition(state,'RESERVE')
        self.assertEqual(audit['verdict'],'UNKNOWN');self.assertEqual(after,saved)
    def test_no_internal_provenance_in_policy_prompt(self):
        env,state=prepare('query','OPEN_VAULT FETCH_COUNTS')
        for m in env.observation(state)['authorized_receipts']:
            self.assertEqual(set(m),{'event_id','sender','recipient','content'})
        self.assertTrue(state.access_log)
    def test_resealed_malformed_table_is_rejected_without_exception(self):
        env,state=prepare('query','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES SEND_CITIES')
        message=next(m for m in state.ledger if m['content']['type']=='counts')
        message['content']['rows'][0]['company_id']=[];message['sha256']=env.evidence_hash(message)
        after,audit=env.transition(state,'JOIN')
        self.assertEqual(audit['verdict'],'UNKNOWN');self.assertEqual(after,state)


if __name__=='__main__':unittest.main()
