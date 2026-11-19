"""CCE training-credit contract: exact synthetic Q* is training-only."""
from __future__ import annotations
from dataclasses import replace
import math,random,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,UniformPolicy,sample_group,Decision,Episode
from kci.agent_rl.cce import NativeCCEOracle,score_credit,native_future_value
from kci.agent_rl.grpo import GRPOUpdater,GRPOConfig
try:
    import torch
except ImportError:
    torch=None

class CCEContractTests(unittest.TestCase):
    def test_truthful_early_credits(self):
        world=Scenario("meeting",False,False,11)
        episodes=sample_group(world,UniformPolicy(),8,72043)
        oracle=NativeCCEOracle()
        credits,proofs,stats=oracle.make_credits(world,episodes)
        self.assertEqual(len(credits),8)
        self.assertTrue(proofs)
        self.assertGreater(stats["nonzero_credits"],0)
        self.assertGreaterEqual(stats["exact_query_state_calls"],1)
        for p in proofs:
            self.assertIn(p["role"],("Planner","Broker","HR"))
            self.assertTrue(p["exclude_oracle_from_inference"])
            self.assertEqual(p["teacher"],"native_synthetic_exact_Qstar")
            self.assertEqual({v["legal"] for v in p["alternatives"].values()},{True})
            self.assertAlmostEqual(abs(p["score"]),1.,delta=.001)
            prompt=episodes[p["sample_index"]].decisions[p["decision_index"]].prompt
            self.assertNotIn("optimal_terminal_turn",prompt)
            self.assertNotIn("safe_success_reachable",prompt)

    def test_same_prefix_hash_mismatch_fails(self):
        world=Scenario("meeting",True,False,11)
        group=sample_group(world,UniformPolicy(),2,171)
        group[0].decisions[0].state_hash="corrupted"
        with self.assertRaises(RuntimeError):
            NativeCCEOracle().make_credits(world,group)

    def test_historical_condition_and_action_status(self):
        w=Scenario("meeting",True,False,11).build()
        s=w.decision_start()
        out={a:native_future_value(w,s,a) for a in w.options(s)}
        self.assertEqual(len(out),2)
        self.assertTrue(all(x["legal"] for x in out.values()))
        self.assertEqual({x["reachable"] for x in out.values()},{True})

    def test_blending_is_bounded_and_missing_local_preserves_terminal(self):
        self.assertEqual(score_credit(.7,None,.6),.7)
        self.assertAlmostEqual(score_credit(.7,-1,.5),-.15)
        with self.assertRaises(ValueError):score_credit(.7,2.0,.5)
        with self.assertRaises(ValueError):score_credit(.7,0,-.2)

@unittest.skipIf(torch is None,"PyTorch unavailable")
class CCEGradientTests(unittest.TestCase):
    class MiniPolicy:
        trainable=True
        policy_id="cce-mini"
        def __init__(self):
            self.torch=torch
            self.model=torch.nn.Linear(1,2)
            with torch.no_grad():
                self.model.weight.zero_();self.model.bias.zero_()
        def trainable_parameters(self):return list(self.model.parameters())
        def log_distribution(self,prompt,actions):
            logits=self.model(torch.ones(1,1))[0,:len(actions)]
            return torch.log_softmax(logits,dim=-1)
        def reference_log_distribution(self,prompt,actions):
            return torch.log_softmax(torch.zeros(len(actions)),dim=-1)
    @staticmethod
    def ep(i,choice):
        d=Decision(actor="Planner",state_hash="s",observation_hash="o",
            prompt="safe synthetic mock",actions=["SEND_ROSTER","PRIVATE_BOOKING"],
            chosen_index=choice,chosen_action=["SEND_ROSTER","PRIVATE_BOOKING"][choice],
            chosen_letter="AB"[choice],chosen_token_id=choice,
            old_logprob=-math.log(2),entropy=math.log(2),
            verdict="VALID",newly_derived=[],receipt_count_before=0,
            receipt_count_after=1)
        return Episode(scenario_key="same",sample_index=i,rollout_seed=123+i,
            policy_id="cce-mini",split="test",initial_state_hash="identical",
            decisions=[d],success=False,violation=False,done=True,
            total_steps=1,reward=0.,final_receipt_digest="mock")
    def test_local_state_credit_updates_under_zero_terminal_variance(self):
        p=self.MiniPolicy()
        grpo=GRPOUpdater(p,GRPOConfig(lr=.005,kl_coef=0.))
        e=[self.ep(0,0),self.ep(1,1)]
        before=p.log_distribution("mock",("a","b"))[0].exp().item()
        res=grpo.update(e,local_credits=[{0:1.},{0:-1.}],cce_weight=.5)
        after=p.log_distribution("mock",("a","b"))[0].exp().item()
        self.assertTrue(res["updated"])
        self.assertEqual(res["optimizer_step"],1)
        self.assertEqual(res["cce_nonzero_local_decisions"],2)
        self.assertGreater(after,before)
    def test_no_teacher_means_baseline_zero_variance_still_skips(self):
        p=self.MiniPolicy()
        grpo=GRPOUpdater(p,GRPOConfig(lr=.005,kl_coef=0.))
        res=grpo.update([self.ep(0,0),self.ep(1,1)])
        self.assertFalse(res["updated"])
        self.assertEqual(grpo.optimizer_steps,0)

if __name__=="__main__":unittest.main()
