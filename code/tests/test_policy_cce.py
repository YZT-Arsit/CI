"""Policy-conditioned CCE: continuation, no Qstar, terminal-credit retention."""
from __future__ import annotations
from dataclasses import replace
import math,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,UniformPolicy,sample_group,Decision,Episode
from kci.agent_rl.policy_cce import PolicyCCEEstimator
from kci.agent_rl.grpo import GRPOConfig,GRPOUpdater
try:import torch
except ImportError:torch=None

class PolicyCCETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario=Scenario("meeting",True,False,11)
        cls.policy=UniformPolicy()
        cls.group=sample_group(cls.scenario,cls.policy,8,71214)
    def test_frozen_rollout_evidence_contract(self):
        a=PolicyCCEEstimator(n_continuations=4,min_mean_gap=.25,
                             min_consensus=.6,max_prefixes_per_group=4)
        credits,records,stats=a.make_credits(self.scenario,self.group,self.policy,group_index=2)
        self.assertEqual(len(credits),8)
        self.assertGreater(stats["queried_prefixes"],0)
        self.assertLessEqual(stats["queried_prefixes"],4)
        self.assertEqual(stats["counterfactual_rollouts"],8*stats["queried_prefixes"])
        self.assertEqual(stats["credit_decisions"],len(records))
        for r in records:
            self.assertEqual(r["teacher_kind"],"frozen_same_step_policy_Qpi")
            self.assertTrue(r["no_native_Qstar_used"])
            self.assertFalse(any(v["qstar"] is not None for v in r["action_returns"]["actions"].values()))
            self.assertGreater(abs(r["local_credit"]),0)
            self.assertTrue(-1<=r["local_credit"]<=1)
            self.assertEqual(r["policy_id"],self.policy.policy_id)
    def test_deterministic_under_fixed_policy_and_seed(self):
        conf=dict(n_continuations=4,min_mean_gap=.25,min_consensus=.6,max_prefixes_per_group=4)
        x=PolicyCCEEstimator(**conf).make_credits(self.scenario,self.group,self.policy,group_index=8)
        y=PolicyCCEEstimator(**conf).make_credits(self.scenario,self.group,self.policy,group_index=8)
        self.assertEqual(x,y)
    def test_stale_policy_id_is_rejected(self):
        bad=[replace(e,policy_id="old-policy") for e in self.group]
        with self.assertRaises(ValueError):
            PolicyCCEEstimator().make_credits(self.scenario,bad,self.policy,group_index=0)
    def test_corrupted_real_agent_prefix_is_rejected(self):
        original=self.group[0].decisions[0].state_hash
        self.group[0].decisions[0].state_hash="corrupted"
        try:
            with self.assertRaises(RuntimeError):
                PolicyCCEEstimator().make_credits(self.scenario,self.group,self.policy,group_index=0)
        finally:self.group[0].decisions[0].state_hash=original
    def test_invalid_confidence_gate_rejected(self):
        for kw in ({"n_continuations":1},{"min_consensus":.5},
                   {"min_mean_gap":0},{"max_prefixes_per_group":0}):
            with self.assertRaises(ValueError):
                PolicyCCEEstimator(**kw)

@unittest.skipIf(torch is None,"PyTorch unavailable")
class AdditiveGradientTests(unittest.TestCase):
    class Mock:
        trainable=True
        policy_id="mock"
        def __init__(self):
            self.torch=torch;self.model=torch.nn.Linear(1,2)
            with torch.no_grad():self.model.weight.zero_();self.model.bias.zero_()
        def trainable_parameters(self):return list(self.model.parameters())
        def log_distribution(self,prompt,actions):
            return torch.log_softmax(self.model(torch.ones(1,1))[0,:len(actions)],dim=0)
        def reference_log_distribution(self,prompt,actions):
            return torch.log_softmax(torch.zeros(len(actions)),dim=0)
    @staticmethod
    def make_ep(i,choice,reward):
        d=Decision(actor="Planner",state_hash="s",observation_hash="o",
                   prompt="test",actions=["SAFE_A","SAFE_B"],
                   chosen_index=choice,chosen_action=["SAFE_A","SAFE_B"][choice],
                   chosen_letter="AB"[choice],chosen_token_id=choice,
                   old_logprob=-math.log(2),entropy=math.log(2),
                   verdict="VALID",newly_derived=[],receipt_count_before=0,
                   receipt_count_after=1)
        return Episode(scenario_key="same",sample_index=i,rollout_seed=i,
           policy_id="mock",split="train",initial_state_hash="same",
           decisions=[d],success=reward>0,violation=False,done=True,
           total_steps=1,reward=float(reward),final_receipt_digest="toy")
    def test_additive_no_local_is_identical_to_terminal_grpo(self):
        a,b=self.Mock(),self.Mock()
        group=[self.make_ep(0,0,1),self.make_ep(1,1,0)]
        cfg=GRPOConfig(lr=.005,kl_coef=0.)
        GRPOUpdater(a,cfg).update(group)
        GRPOUpdater(b,cfg).update(group,local_credits=[{},{}],
                                  cce_weight=.5,credit_style="additive")
        self.assertTrue(torch.allclose(a.model.bias,b.model.bias,atol=1e-8))
    def test_additive_local_changes_zero_variance_group(self):
        p=self.Mock()
        group=[self.make_ep(0,0,0),self.make_ep(1,1,0)]
        before=p.log_distribution("test",("A","B"))[0].exp().item()
        o=GRPOUpdater(p,GRPOConfig(lr=.005,kl_coef=0.))
        report=o.update(group,local_credits=[{0:.5},{0:-.5}],
                        cce_weight=.5,credit_style="additive")
        after=p.log_distribution("test",("A","B"))[0].exp().item()
        self.assertTrue(report["updated"])
        self.assertGreater(after,before)
        self.assertEqual(report["credit_style"],"additive")
    def test_unrecognized_credit_style_rejected(self):
        p=self.Mock()
        group=[self.make_ep(0,0,1),self.make_ep(1,1,0)]
        with self.assertRaises(ValueError):
            GRPOUpdater(p,GRPOConfig()).update(group,credit_style="unknown")
if __name__=="__main__":unittest.main()
