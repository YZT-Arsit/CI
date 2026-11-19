"""Native rollout auditor must detect any forged action, prompt, reward or receipt."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,UniformPolicy,execute_episode
from experiments.receipt_dataset import SOURCE

spec=importlib.util.spec_from_file_location("kci_agent_auditor",ROOT/"tools"/"audit_kci_agent_rl.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class NativeReplayAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode=execute_episode(
            Scenario("meeting",True,False,11),UniformPolicy(),0,7341).serialize()

    def test_true_rollout_replays_byte_for_byte(self):
        self.assertIsNotNone(audit.audit_episode(self.episode,SOURCE))

    def test_wrong_action_rejected(self):
        bad=deepcopy(self.episode)
        d=bad["decisions"][0]
        d["chosen_action"]="NONEXISTENT_ACTION"
        with self.assertRaises(AssertionError):
            audit.audit_episode(bad,SOURCE)

    def test_wrong_reward_rejected(self):
        bad=deepcopy(self.episode)
        bad["reward"]=123.
        with self.assertRaises(AssertionError):
            audit.audit_episode(bad,SOURCE)

    def test_modified_agent_prompt_rejected(self):
        bad=deepcopy(self.episode)
        bad["decisions"][0]["prompt"]+=" hidden answer"
        with self.assertRaises(AssertionError):
            audit.audit_episode(bad,SOURCE)

    def test_forged_receipt_rejected(self):
        bad=deepcopy(self.episode)
        bad["final_receipt_digest"]="bad-hash"
        with self.assertRaises(AssertionError):
            audit.audit_episode(bad,SOURCE)

    def test_mismatched_source_world_rejected(self):
        bad=deepcopy(self.episode)
        bad["scenario_key"]="meeting_related0_allowed0_h11"
        with self.assertRaises(AssertionError):
            audit.audit_episode(bad,SOURCE)

    def test_invalid_scenario_key_fails(self):
        with self.assertRaises(ValueError):
            audit.scenario_for("patient_real_private_2026")

if __name__=="__main__":unittest.main()
