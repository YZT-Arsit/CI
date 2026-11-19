"""Independent frozen-policy Qpi fork auditing and fixed cohort tests."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import importlib.util,sys,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,UniformPolicy
from kci.agent_rl.policy_continuation import (
    make_fork,frozen_policy_counterfactual,compare_oracle_policy
)
spec=importlib.util.spec_from_file_location(
    "policy_qpi_independent",ROOT/"tools"/"audit_kci_policy_qpi.py")
qa=importlib.util.module_from_spec(spec);spec.loader.exec_module(qa)

SELECT=ROOT/"artifacts"/"20261119_composition_signals"/"agent_rl_v1"/"policy_qpi_01"/"selection_v2"

class FrozenModelAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fork=make_fork(Scenario("meeting",False,False,11),())
        cls.item=frozen_policy_counterfactual(
            fork,UniformPolicy(),n_continuations=3,seed=41)
        cls.item["comparison"]=compare_oracle_policy(cls.item)
    def test_native_replay_accepts_trustworthy_Qpi_result(self):
        self.assertTrue(qa.audit_record(self.item,None))
    def test_tampered_future_action_is_caught(self):
        r=deepcopy(self.item)
        r["actions"][sorted(r["actions"])[0]]["rollouts"][0]["continuation_actions"][0]="ILLEGAL"
        with self.assertRaises(AssertionError):
            qa.audit_record(r,None)
    def test_tampered_terminal_reward_is_caught(self):
        r=deepcopy(self.item)
        r["actions"][sorted(r["actions"])[0]]["rollouts"][0]["reward"]=+99.
        with self.assertRaises(AssertionError):
            qa.audit_record(r,None)
    def test_tampered_oracle_qstar_is_caught(self):
        r=deepcopy(self.item)
        r["actions"][sorted(r["actions"])[0]]["qstar"]["q"]=.123
        with self.assertRaises(AssertionError):
            qa.audit_record(r,None)
    def test_forged_receipt_hash_is_caught(self):
        r=deepcopy(self.item)
        r["actions"][sorted(r["actions"])[0]]["rollouts"][0]["final_receipts_hash"]="fake"
        with self.assertRaises(AssertionError):
            qa.audit_record(r,None)

class CohortContractTests(unittest.TestCase):
    def test_selection_groups_replay_and_diff_only_history(self):
        selection=qa.read(SELECT/"SELECTION.json")
        allforks=qa.rows(SELECT/"FORKS.jsonl")
        self.assertEqual(len(allforks),16)
        self.assertEqual(selection["selected_history_pairs"],8)
        self.assertEqual(selection["selected_history_sensitive"],4)
        self.assertEqual(selection["n_rollouts_per_action"],8)
        for i in range(0,len(allforks),2):
            a,b=allforks[i:i+2]
            self.assertEqual((a["pair_id"],b["pair_id"]),(i//2,i//2))
            aa,bb=a["scenario"],b["scenario"]
            self.assertEqual((aa["kind"],aa["allow"],aa["horizon"]),
                             (bb["kind"],bb["allow"],bb["horizon"]))
            self.assertFalse(aa["related"])
            self.assertTrue(bb["related"])
            self.assertEqual(a["prefix"],b["prefix"])
            self.assertEqual((a["role"],a["phase"]),(b["role"],b["phase"]))

if __name__=="__main__":unittest.main()
