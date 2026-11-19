"""Tests for real same-state policy-conditioned CCE continuation.

No model weights/GPU; deterministic frozen policy and native executable world.
"""
from __future__ import annotations
from dataclasses import asdict,replace
import math,random,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import ActionSample,Scenario,UniformPolicy,digest
from kci.agent_rl.policy_continuation import (
 FrozenFork,replay_prefix,make_fork,immediately_legal,frozen_continuation,
 frozen_policy_counterfactual,compare_oracle_policy,sign_with_margin
)
class RecordingPolicy:
    policy_id="frozen-policy-unchanged"
    def __init__(self):
        self.prompts=[]
    def choose(self,prompt,actions,rng):
        self.prompts.append(prompt)
        assert "optimal_terminal_turn" not in prompt
        assert "true_relation" not in prompt
        assert "safe_success_reachable" not in prompt
        return ActionSample(index=rng.randrange(len(actions)),
                            logprob=-math.log(len(actions)),
                            entropy=math.log(len(actions)))

class PolicyContinuationTests(unittest.TestCase):
    def test_initial_fork_observation_invariant(self):
        s=Scenario("meeting",False,False,11)
        f=make_fork(s,())
        self.assertEqual(f.phase,"choose")
        self.assertEqual(f.role,"Planner")
        env,state=replay_prefix(s,())
        self.assertEqual(digest(asdict(state)),f.state_hash)
        self.assertEqual(len(immediately_legal(env,state)),2)
        self.assertNotIn("success",env.observation(state))

    def test_forced_candidate_and_policy_continuation_truth(self):
        s=Scenario("meeting",True,False,11)
        env,state=replay_prefix(s,())
        p=RecordingPolicy()
        values=[]
        for a in immediately_legal(env,state):
            r=frozen_continuation(env,state,a,p,711)
            self.assertEqual(r["forced_action"],a)
            self.assertEqual(r["continuation_actions"][0],a)
            self.assertEqual(r["policy_id"],p.policy_id)
            self.assertTrue(r["safe_success"] or r["ci_violation"] or r["safe_failure"])
            self.assertEqual(sum([r["safe_success"],r["ci_violation"],r["safe_failure"]]),1)
            values.append(r)
        self.assertEqual(state.turn,1,"Counterfactual rollout cannot mutate original fork")
        self.assertTrue(p.prompts)

    def test_paired_common_random_numbers_reproducible(self):
        s=Scenario("query",False,False,11)
        f=make_fork(s,())
        a=frozen_policy_counterfactual(f,UniformPolicy(),n_continuations=4,seed=144)
        b=frozen_policy_counterfactual(f,UniformPolicy(),n_continuations=4,seed=144)
        self.assertEqual(a,b)
        self.assertEqual(len(a["actions"]),2)
        self.assertEqual(sum(len(v["rollouts"]) for v in a["actions"].values()),8)
        self.assertTrue(a["no_oracle_in_model_observations"])
        compare=compare_oracle_policy(a)
        self.assertIn(compare["ranking"],("agree","conflict","policy_tie","teacher_tie"))

    def test_corrupt_prefix_and_state_are_rejected(self):
        s=Scenario("query",True,False,11)
        with self.assertRaises(ValueError):
            make_fork(s,("NOT_A_REAL_ACTION",))
        f=make_fork(s,())
        with self.assertRaises(RuntimeError):
            frozen_policy_counterfactual(replace(f,state_hash="bad"),
                                          UniformPolicy(),n_continuations=2)

    def test_frozen_policy_id_must_not_change(self):
        class Mutating(RecordingPolicy):
            def choose(self,prompt,actions,rng):
                self.policy_id="changed-during-counterfactual"
                return super().choose(prompt,actions,rng)
        f=make_fork(Scenario("meeting",True,True,11),())
        with self.assertRaises(RuntimeError):
            frozen_policy_counterfactual(f,Mutating(),n_continuations=2)

    def test_excluded_abort_and_permitted_actions(self):
        s=Scenario("query",True,False,11)
        env,state=replay_prefix(s,("OPEN_VAULT",))
        actions=immediately_legal(env,state)
        self.assertNotIn("ABORT",actions)
        self.assertGreaterEqual(len(actions),2)

    def test_model_qpi_not_equal_qstar_by_construction(self):
        f=make_fork(Scenario("meeting",True,False,11),())
        r=frozen_policy_counterfactual(f,UniformPolicy(),n_continuations=4)
        for v in r["actions"].values():
            self.assertIsInstance(v["mean_policy_reward"],float)
            self.assertTrue(v["qstar"] is not None)
            self.assertEqual(v["safe_successes"]+v["violations"]+v["safe_failures"],4)
        self.assertTrue(r["qstar_is_ideal_not_expected_return"])

    def test_margin_comparison_is_explicit(self):
        self.assertEqual(sign_with_margin(.2,.19,.02),0)
        self.assertEqual(sign_with_margin(.3,.2,.02),1)
        self.assertEqual(sign_with_margin(.2,.3,.02),-1)
        with self.assertRaises(ValueError):
            sign_with_margin(0,0,-1)

if __name__=="__main__":unittest.main()
