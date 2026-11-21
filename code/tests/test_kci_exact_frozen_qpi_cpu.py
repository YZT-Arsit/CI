"""CPU exact Qpi source/native policy invariants; no new task fixtures."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
import sys,unittest,math
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_exact_frozen_qpi_cpu import (
 ROWS,GROUP,environment,prepare_A,ExactQpi,load_frozen_policy,qstar_A_from_original)
from chain_shared_actor_cpu_action_interaction import (
 valid_decision_candidates,actor_feature,score_actions,digest)
from solve_v31_longterm import native_best

class ExactPolicyQpi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.meta=load_frozen_policy("action_sft",7)

    def test_checkpoint_frozen_and_no_dev_or_test_in_train(self):
        assert self.meta["checkpoint_sha256"]
        source={r["graph"] for r in GROUP["train"]}
        other={r["graph"] for r in GROUP["dev"]+GROUP["exploratory_test"]}
        self.assertTrue(source.isdisjoint(other))
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))
        self.assertTrue(all(p.device.type=='cpu' for p in self.model.parameters()))

    def test_same_signed_A_source_state_legal_fork(self):
        row=ROWS["DG19_M01__tradeoff__swap0"]
        env=environment(row)
        prior=prepare_A(env)
        h=digest(asdict(prior))
        opts=env.options(prior)
        self.assertIn("A_SEND_PACKAGE_A",opts)
        self.assertIn("A_SEND_PACKAGE_B",opts)
        first,recA=env.transition(prior,"A_SEND_PACKAGE_A")
        second,recB=env.transition(prior,"A_SEND_PACKAGE_B")
        self.assertEqual(digest(asdict(prior)),h)
        self.assertEqual(recA["verdict"],recB["verdict"])
        self.assertTrue(all(x["sender"]=="RecordsAgent" for x in first.native.ledger+second.native.ledger))
        self.assertNotEqual(first.native.delivered["R"],second.native.delivered["R"])
        self.assertFalse(first.native.violation or second.native.violation)

    def test_exact_frozen_Qpi_reproduces_DG19_conflict_from_native(self):
        row=ROWS["DG19_M01__tradeoff__swap0"]
        env=environment(row)
        src=prepare_A(env)
        checksum=sha256(str([(k,v.detach().clone().tolist()) for k,v in
                      self.model.state_dict().items()]).encode("utf-8")).hexdigest()
        qpi=ExactQpi(env,self.model,self.meta["checkpoint_sha256"]).evaluate_A_arms(src)
        self.assertAlmostEqual(qpi["Qpi_A_minus_B"],-0.048719277,places=6)
        self.assertAlmostEqual(qpi["arms"]["PACKAGE_A"]["Qpi"],0.516210294,places=6)
        self.assertAlmostEqual(qpi["arms"]["PACKAGE_B"]["Qpi"],0.564929571,places=6)
        self.assertAlmostEqual(qpi["arms"]["PACKAGE_A"]["CI_violation_probability"],0.067304205,places=6)
        self.assertGreater(qpi["cost"]["native_states"],50)
        self.assertGreater(qpi["cost"]["actual_native_transition_edges"],50)
        self.assertEqual(len(qpi["arms"]["PACKAGE_A"]["actual_A_delivered_ledger"]),2)
        vals=qstar_A_from_original(row,verify=True)
        self.assertAlmostEqual(vals["PACKAGE_A"]-vals["PACKAGE_B"],0.028,places=6)
        after=sha256(str([(k,v.detach().clone().tolist()) for k,v in
                      self.model.state_dict().items()]).encode("utf-8")).hexdigest()
        self.assertEqual(checksum,after)

    def test_full_unforced_policy_Qpi_equals_A_choice_mixture(self):
        row=ROWS["DG19_M01__tradeoff__swap0"]
        env=environment(row)
        p=ExactQpi(env,self.model,self.meta["checkpoint_sha256"])
        astate=prepare_A(env)
        arms=p.evaluate_A_arms(astate)
        obs=env.observation(astate)
        actions=valid_decision_candidates(obs,env.options(astate))
        probs=p.probs(obs,actions)
        pmap=dict(zip(actions,probs))
        self.assertIn("ABORT_CHAIN",pmap)
        explicit=(pmap["A_SEND_PACKAGE_A"]*arms["arms"]["PACKAGE_A"]["Qpi"]+
                  pmap["A_SEND_PACKAGE_B"]*arms["arms"]["PACKAGE_B"]["Qpi"])
        native_all=p.walk(env.initial())
        self.assertAlmostEqual(explicit,native_all[0],places=6)
        self.assertAlmostEqual(sum(native_all[1:4]),1.,places=6)

    def test_B_source_role_and_native_truth_are_checked(self):
        row=ROWS["DG19_M01__tradeoff__swap0"]
        env=environment(row)
        s=prepare_A(env)
        for p in ("PACKAGE_A","PACKAGE_B"):
            state,rec=env.transition(s,"A_SEND_"+p)
            self.assertEqual(rec["verdict"],"VALID")
            for a in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                state,rec=env.transition(state,a)
                self.assertEqual(rec["verdict"],"VALID")
            obs=env.observation(state)
            self.assertEqual(obs["actor"],"Coordinator")
            self.assertEqual(obs["current_stage"],"preflight")
            for name in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"):
                self.assertIn(name,obs["legal_actions"])
            self.assertFalse(any(k in obs for k in ("Qstar","Qpi","correct_action","true_return")))

if __name__=="__main__":unittest.main()
