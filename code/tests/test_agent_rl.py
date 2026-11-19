"""KCI on-policy RL execution invariants; all tests run without torch/GPU."""
from __future__ import annotations
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import (
    ActionSample, UniformPolicy, aggregate, execute_episode, group_advantages,
    prompt_for, sample_group, scenarios, terminal_reward, Scenario
)
from experiments.receipt_dataset import SOURCE


class BadPolicy:
    policy_id="out_of_contract"
    def choose(self,prompt,actions,rng):
        return ActionSample(index=100,logprob=-1.,entropy=0.)


class RLExecutionTests(unittest.TestCase):
    def setUp(self):
        self.world=Scenario("meeting",True,False,11)
        self.policy=UniformPolicy()

    def test_grid_separated_by_horizon_only(self):
        train,dev=scenarios("train"),scenarios("dev")
        self.assertEqual((len(train),len(dev)),(32,32))
        self.assertFalse({x.key for x in train}&{x.key for x in dev})
        self.assertEqual({x.kind for x in train},{"meeting","query"})
        self.assertEqual({x.kind for x in dev},{"meeting","query"})

    def test_native_complete_rollouts_and_no_unknown(self):
        for kind in ("meeting","query"):
            ep=execute_episode(Scenario(kind,True,False,11),
                    self.policy,0,3001)
            self.assertTrue(ep.done)
            self.assertLessEqual(ep.total_steps,11)
            self.assertTrue(ep.decisions)
            self.assertFalse(ep.success and ep.violation)
            self.assertTrue(all(d.chosen_action in d.actions for d in ep.decisions))
            self.assertTrue(all(d.verdict in ("VALID","VIOLATION") for d in ep.decisions))
            self.assertTrue(all(d.receipt_count_after>=d.receipt_count_before for d in ep.decisions))

    def test_action_observations_no_private_oracle(self):
        env=self.world.build()
        s=env.decision_start()
        obs=env.observation(s)
        text=prompt_for(obs,tuple(env.options(s)))
        self.assertIn("authorized_receipts",text)
        self.assertIn("EXECUTABLE_OPTIONS:",text)
        self.assertNotIn('"related":',text)
        self.assertNotIn('"success":',text)
        self.assertNotIn('"violation":',text)
        self.assertNotIn('"expected":',text)

    def test_fake_truth_and_changed_action_set_rejected(self):
        env=self.world.build();s=env.decision_start();ob=env.observation(s)
        for field in ("success","violation","related","expected"):
            with self.assertRaises(ValueError):
                prompt_for({**ob,field:"leak"},tuple(env.options(s)))
        with self.assertRaises(ValueError):
            prompt_for(ob,tuple(reversed(env.options(s))))

    def test_model_cannot_invent_action(self):
        with self.assertRaises(RuntimeError):
            execute_episode(self.world,BadPolicy(),0,5)

    def test_replayed_episode_identical(self):
        x=execute_episode(self.world,self.policy,0,1234).serialize()
        y=execute_episode(self.world,self.policy,0,1234).serialize()
        self.assertEqual(x,y)

    def test_same_prefix_group_and_gradient_evidence(self):
        group=sample_group(self.world,self.policy,8,4711)
        self.assertEqual(len({e.initial_state_hash for e in group}),1)
        self.assertEqual(len(group),8)
        if len({e.reward for e in group})>1:
            a=group_advantages(group)
            self.assertEqual(len(a),8)
            self.assertLess(abs(sum(a)),1e-6)
        score=aggregate([group])
        self.assertEqual(score["rollouts"],8)
        self.assertEqual(score["groups"],1)

    def test_mixed_group_forbidden(self):
        a=execute_episode(self.world,self.policy,0,1)
        b=execute_episode(Scenario("meeting",False,False,11),
                           self.policy,1,2)
        with self.assertRaises(ValueError):
            group_advantages([a,b])

    def test_zero_variance_group_skipped(self):
        a=execute_episode(self.world,self.policy,0,999)
        b=replace(a,sample_index=1)
        self.assertIsNone(group_advantages([a,b]))

    def test_reward_ordering_and_no_faked_success(self):
        self.assertEqual(terminal_reward(success=True,violation=True,done=True,steps=4),-1.)
        self.assertEqual(terminal_reward(success=False,violation=False,done=True,steps=8),0.)
        self.assertGreater(terminal_reward(success=True,violation=False,done=True,steps=9),0.)
        self.assertLess(terminal_reward(success=False,violation=True,done=True,steps=3),0.)

    def test_uniform_model_logprobs_match_choice_support(self):
        for n in (1,2,3,5):
            x=self.policy.choose("p",tuple(str(i) for i in range(n)),random.Random(123))
            self.assertTrue(0<=x.index<n)
            self.assertAlmostEqual(x.logprob,-math.log(n))
            self.assertAlmostEqual(x.entropy,math.log(n))

    def test_world_input_contract_rejected(self):
        with self.assertRaises(ValueError):Scenario("fictional",False,False,11)
        with self.assertRaises(ValueError):Scenario("meeting",0,False,11)
        with self.assertRaises(ValueError):Scenario("meeting",True,False,3)
        with self.assertRaises(ValueError):scenarios("test")


if __name__=="__main__":
    unittest.main()
