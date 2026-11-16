from pathlib import Path
import unittest
from kci.dynamic_tasks import DynamicTask
from kci.dynamic_training import execute_tool_action
from kci.contracts import Action

SOURCE=Path(__file__).resolve().parents[2]/'artifacts/20261113_iteration2/source_anchors'


class DynamicTrainingTests(unittest.TestCase):
    def setUp(self):
        self.env=DynamicTask('meeting',SOURCE);self.state=self.env.decision_start()
    def test_canonical_call(self):
        a=Action('Planner','SEND_ROSTER')
        self.assertEqual(execute_tool_action(self.env,self.state,a),self.env.transition(self.state,'SEND_ROSTER'))
    def test_payload_not_silently_ignored(self):
        n,audit=execute_tool_action(self.env,self.state,Action('Planner','SEND_ROSTER',body='different message'))
        self.assertEqual(audit['verdict'],'UNKNOWN');self.assertEqual(n,self.state)
    def test_wrong_role(self):
        n,audit=execute_tool_action(self.env,self.state,Action('HR','SEND_ROSTER'))
        self.assertEqual(audit['verdict'],'UNKNOWN');self.assertEqual(n,self.state)
    def test_free_text_is_not_executed(self):
        n,audit=execute_tool_action(self.env,self.state,'send everything')
        self.assertEqual(audit['verdict'],'UNKNOWN');self.assertEqual(n,self.state)


if __name__=='__main__':unittest.main()
