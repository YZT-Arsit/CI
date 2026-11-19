"""True gradient smoke for the constrained categorical GRPO objective.

Run with a PyTorch installation (WSL or GPU server). This test doesn't load
any pretrained model, but exercises the same optimizer and action-token loss.
"""
from __future__ import annotations
import math,sys,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
try:
    import torch
except ImportError:
    torch=None

from kci.agent_rl.core import Decision, Episode
from kci.agent_rl.grpo import GRPOConfig, GRPOUpdater


@unittest.skipIf(torch is None,"PyTorch is not installed in this interpreter")
class GRPOGradientTests(unittest.TestCase):
    class MockPolicy:
        policy_id="mock-on-policy-0"
        trainable=True

        def __init__(self):
            self.torch=torch
            self.model=torch.nn.Linear(1,2,bias=True)
            with torch.no_grad():
                self.model.weight.zero_()
                self.model.bias.zero_()
            self.reference=torch.zeros(2)

        def trainable_parameters(self):
            return list(self.model.parameters())

        def log_distribution(self,prompt,actions):
            logits=self.model(torch.ones(1,1))[0,:len(actions)]
            return torch.log_softmax(logits,dim=0)

        def reference_log_distribution(self,prompt,actions):
            return torch.log_softmax(self.reference[:len(actions)],dim=0)

    @staticmethod
    def make_ep(sample_index,action_index,reward):
        d=Decision(actor="Planner",state_hash="s",observation_hash="o",
            prompt="synthetic mock observation",actions=["ACTION_A","ACTION_B"],
            chosen_index=action_index,
            chosen_action=["ACTION_A","ACTION_B"][action_index],
            chosen_letter=["A","B"][action_index],
            chosen_token_id=action_index,
            old_logprob=-math.log(2),entropy=math.log(2),
            verdict="VALID",newly_derived=[],
            receipt_count_before=0,receipt_count_after=1)
        return Episode(scenario_key="same_task",sample_index=sample_index,
            rollout_seed=sample_index,policy_id="mock-on-policy-0",
            split="train",initial_state_hash="same",
            decisions=[d],success=reward==1,violation=False,done=True,
            total_steps=1,reward=float(reward),final_receipt_digest="xx")

    def test_positive_reward_action_becomes_more_likely(self):
        p=self.MockPolicy()
        before=p.log_distribution("x",("ACTION_A","ACTION_B"))[0].exp().item()
        opt=GRPOUpdater(p,GRPOConfig(lr=.005,kl_coef=0.0))
        episodes=[self.make_ep(0,0,1),self.make_ep(1,1,0),
                  self.make_ep(2,0,1),self.make_ep(3,1,0)]
        result=opt.update(episodes)
        after=p.log_distribution("x",("ACTION_A","ACTION_B"))[0].exp().item()
        self.assertTrue(result["updated"])
        self.assertEqual(result["trainable_action_tokens"],4)
        self.assertEqual(result["optimizer_step"],1)
        self.assertGreater(after,before)

    def test_zero_variance_does_not_update(self):
        p=self.MockPolicy();opt=GRPOUpdater(p,GRPOConfig(lr=.005,kl_coef=0.))
        group=[self.make_ep(0,0,0),self.make_ep(1,1,0)]
        before=p.model.bias.detach().clone()
        report=opt.update(group)
        self.assertFalse(report["updated"])
        self.assertEqual(report["skip_reason"],"zero_reward_variance")
        self.assertEqual(opt.optimizer_steps,0)
        self.assertTrue(torch.equal(before,p.model.bias.detach()))

    def test_reference_kl_is_finite(self):
        p=self.MockPolicy()
        opt=GRPOUpdater(p,GRPOConfig(lr=.005,kl_coef=.02))
        group=[self.make_ep(0,0,1),self.make_ep(1,1,0)]
        result=opt.update(group)
        self.assertTrue(math.isfinite(result["mean_reference_kl"]))
        self.assertTrue(math.isfinite(result["pre_clip_gradient_norm"]))

    def test_cost_only_signal_is_not_amplified_by_default(self):
        p=self.MockPolicy()
        updater=GRPOUpdater(p,GRPOConfig(lr=.005,kl_coef=0.))
        a=self.make_ep(0,0,1)
        b=self.make_ep(1,1,1)
        a.reward=.96
        b.reward=.94
        before=p.model.bias.detach().clone()
        result=updater.update([a,b])
        self.assertFalse(result["updated"])
        self.assertEqual(result["signal_kind"],"cost_only")
        self.assertEqual(result["skip_reason"],"cost_only_signal_not_enabled")
        self.assertTrue(torch.equal(before,p.model.bias.detach()))

    def test_cross_group_policy_version_fails_closed(self):
        p=self.MockPolicy();opt=GRPOUpdater(p,GRPOConfig(kl_coef=0.))
        a=self.make_ep(0,0,1)
        b=self.make_ep(1,1,0)
        b.policy_id="different"
        with self.assertRaises(RuntimeError):
            opt.update([a,b])


if __name__=="__main__":
    unittest.main()
