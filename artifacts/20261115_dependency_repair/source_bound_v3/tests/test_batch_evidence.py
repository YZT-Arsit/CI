import json
from pathlib import Path
import tempfile
import unittest

from kci.batch_evidence import ComparisonIdentity, EvidenceBudgetPlan, BudgetedPairedComparator
from kci.continuation import DeterministicPolicy
from kci.toy_env import ToyCIEnv
from examples.run_demo import bad_current_policy


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.tmp.name) / 'ledger.jsonl'
        self.env = ToyCIEnv()
        self.state = self.env.initial()
        self.a = self.env.act('A', 'SKIP')
        self.b = self.env.act('A', 'SEND', 'R', ('identity',))
        self.policy = DeterministicPolicy(bad_current_policy, 'frozen_test_control')
        self.identity = ComparisonIdentity.from_context('pair0', self.env, self.policy,
                                                       self.state, self.a, self.b)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, looks=(32, 64)):
        return EvidenceBudgetPlan((self.identity,), looks, .05)

    def compare(self, runner, n=64):
        return runner.compare('pair0', n, self.env, self.policy, self.state, self.a, self.b)

    def test_existing_comparator_and_preference_binding(self):
        runner = BudgetedPairedComparator(self.plan(), self.ledger)
        result = self.compare(runner)
        self.assertIsNotNone(result.preference)
        self.assertEqual(result.delta, .025)
        self.assertEqual(result.preference.evidence['batch_evidence_plan_id'], runner.plan.plan_id)
        self.assertAlmostEqual(runner.accounting()['spent_delta'], .025)

    def test_restart_cannot_repeat_look(self):
        self.compare(BudgetedPairedComparator(self.plan(), self.ledger))
        reopened = BudgetedPairedComparator(self.plan(), self.ledger)
        with self.assertRaises(ValueError): self.compare(reopened)
        self.compare(reopened, 32)
        self.assertAlmostEqual(reopened.accounting()['spent_delta'], .05)

    def test_unregistered_look_does_not_spend(self):
        runner = BudgetedPairedComparator(self.plan(), self.ledger)
        with self.assertRaises(ValueError): self.compare(runner, 100)
        self.assertEqual(runner.accounting()['reserved_looks'], 0)

    def test_identity_drift_rejected(self):
        runner = BudgetedPairedComparator(self.plan(), self.ledger)
        self.state.cost = 2
        with self.assertRaises(ValueError): self.compare(runner)
        self.state.cost = 0
        self.policy.version = 'changed'
        with self.assertRaises(ValueError): self.compare(runner)
        self.assertEqual(runner.accounting()['reserved_looks'], 0)

    def test_plan_replacement_rejected(self):
        BudgetedPairedComparator(self.plan(), self.ledger)
        with self.assertRaises(ValueError):
            BudgetedPairedComparator(self.plan((64,)), self.ledger)

    def test_failed_attempt_not_refunded(self):
        def broken(obs): raise RuntimeError('fixture interruption')
        self.policy = DeterministicPolicy(broken, 'frozen_test_control')
        runner = BudgetedPairedComparator(self.plan(), self.ledger)
        with self.assertRaises(RuntimeError): self.compare(runner)
        with self.assertRaises(ValueError): self.compare(runner)
        self.assertEqual(runner.accounting()['reserved_looks'], 1)
        self.assertEqual(runner.accounting()['completed_looks'], 0)

    def test_corrupted_ledger_is_rejected(self):
        BudgetedPairedComparator(self.plan(), self.ledger)
        with self.ledger.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'event':'completed','plan_id':self.plan().plan_id,
                               'pair_id':'pair0','samples':64})+'\n')
        with self.assertRaises(ValueError): BudgetedPairedComparator(self.plan(), self.ledger)

    def test_invalid_plans(self):
        for looks in ((64,32),(32,32),(True,),()):
            with self.assertRaises(ValueError): self.plan(looks)
        for delta in (0,1,float('nan')):
            with self.assertRaises(ValueError): EvidenceBudgetPlan((self.identity,), (64,), delta)
        with self.assertRaises(ValueError): EvidenceBudgetPlan((self.identity,self.identity), (64,))


if __name__ == '__main__': unittest.main()
