"""Independent auditor negative tests for state-reset execution and Qpi credits."""
from __future__ import annotations
from copy import deepcopy
import importlib.util,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci.agent_rl.core import Scenario,ActionSample
from kci.agent_rl.policy_continuation import make_fork
from kci.agent_rl.state_restarts import RestartSpec,sample_restart_group
from kci.agent_rl.state_directed_cce import StateDirectedPCCE

spec=importlib.util.spec_from_file_location("state_restart_independent",ROOT/"tools"/"audit_kci_state_restart.py")
auditor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)

class FrozenFirst:
    policy_id="deterministic_first_action_software_test"
    def choose(self,prompt,actions,rng):
        return ActionSample(0,0.,0.)

class RestartAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s=Scenario("meeting",True,False,11)
        cls.spec=RestartSpec(9,s,(),make_fork(s,()).state_hash,"Planner","choose")
        cls.policy=FrozenFirst()
        cls.group=sample_restart_group(cls.spec,cls.policy,8,7712)
        cls.credits,cls.proofs,cls.stats=StateDirectedPCCE(
            n_continuations=4,minimum_gap=.25,minimum_consensus=.6
        ).make_credits(cls.spec,cls.group,cls.policy,group_index=0)
        assert cls.proofs

    def test_real_native_prefix_replays(self):
        ep=self.group[0].serialize()
        self.assertTrue(auditor.verify_native_restart(ep,self.spec,None))

    def test_corrupt_restart_reward_rejected(self):
        ep=self.group[0].serialize()
        ep["reward"]+=5.
        with self.assertRaises(AssertionError):
            auditor.verify_native_restart(ep,self.spec,None)

    def test_tampered_role_authorization_rejected(self):
        ep=self.group[0].serialize()
        ep["decisions"][0]["prompt"]+=" PRIVATE FUTURE ANSWER"
        with self.assertRaises(AssertionError):
            auditor.verify_native_restart(ep,self.spec,None)

    def test_tampered_receipt_digest_rejected(self):
        ep=self.group[0].serialize()
        ep["final_receipt_digest"]="not-native"
        with self.assertRaises(AssertionError):
            auditor.verify_native_restart(ep,self.spec,None)

    def test_real_qpi_credit_fully_recomputes(self):
        proof=self.proofs[0]
        ep=self.group[proof["sample_index"]].serialize()
        cfg={"K":4,"min_return_gap":.25,"min_paired_direction_consensus":.6}
        ans=auditor.audit_credit_proof(proof,ep,self.spec,None,cfg)
        self.assertEqual(ans["role"],"Planner")

    def test_fake_preferred_action_credit_rejected(self):
        p=deepcopy(self.proofs[0])
        p["local_credit"]*=-1.
        ep=self.group[p["sample_index"]].serialize()
        with self.assertRaises(AssertionError):
            auditor.audit_credit_proof(p,ep,self.spec,None,
             {"K":4,"min_return_gap":.25,"min_paired_direction_consensus":.6})

    def test_qstar_label_in_cce_training_proof_rejected(self):
        p=deepcopy(self.proofs[0])
        k=next(iter(p["Qpi_evidence"]["actions"]))
        p["Qpi_evidence"]["actions"][k]["qstar"]={"q":1}
        ep=self.group[p["sample_index"]].serialize()
        with self.assertRaises(AssertionError):
            auditor.audit_credit_proof(p,ep,self.spec,None,
             {"K":4,"min_return_gap":.25,"min_paired_direction_consensus":.6})

if __name__=="__main__":unittest.main()
