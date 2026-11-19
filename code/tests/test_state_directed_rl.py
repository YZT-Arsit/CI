"""Regression tests for state-directed on-policy restarts and CCE training.

All tests require only native environment and optional PyTorch for gradient
sanity; no CUDA/model download/API calls.
"""
from __future__ import annotations
from dataclasses import replace
from collections import Counter
from pathlib import Path
import importlib.util,math,random,sys,unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,ActionSample,UniformPolicy,group_advantages
from kci.agent_rl.state_restarts import (
    RestartSpec,restore,execute_restart_episode,sample_restart_group
)
from kci.agent_rl.state_directed_cce import StateDirectedPCCE,robust_pair_signal
from kci.agent_rl.policy_continuation import make_fork
from kci.agent_rl.grpo import GRPOConfig,GRPOUpdater
try:import torch
except ImportError:torch=None
SELECT=ROOT/"artifacts"/"20261119_composition_signals"/"agent_rl_v1"/"policy_qpi_01"/"selection_v2"

class BrokenPolicy:
    policy_id="bad"
    def choose(self,prompt,actions,rng):
        return ActionSample(index=len(actions)+1,logprob=0.,entropy=0.)

class FirstPolicy:
    policy_id="deterministic_native_first_action"
    def choose(self,prompt,actions,rng):
        return ActionSample(index=0,logprob=0.,entropy=0.)

class StateDirectedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=RestartSpec(17,Scenario("meeting",True,False,11),(),
             make_fork(Scenario("meeting",True,False,11),()).state_hash,
             "Planner","choose")

    def test_all_16_real_restart_prefixes_replay(self):
        import json
        lines=[json.loads(s) for s in (SELECT/"FORKS.jsonl").read_text().splitlines() if s.strip()]
        self.assertEqual(len(lines),16)
        for item in lines:
            s=RestartSpec.from_record(item)
            env,state=restore(s)
            self.assertFalse(state.done)
            self.assertFalse(state.violation)
            self.assertEqual(env.actor(state),s.role)
            self.assertEqual(env.observation(state)["phase"],s.phase)

    def test_sampled_action_is_not_teacher_forced(self):
        ep=execute_restart_episode(self.spec,UniformPolicy(),0,145)
        env,start=restore(self.spec)
        self.assertEqual(ep.decisions[0].state_hash,self.spec.state_hash)
        self.assertEqual(ep.initial_state_hash,self.spec.state_hash)
        self.assertIn(ep.decisions[0].chosen_action,env.options(start))
        self.assertTrue(ep.done)
        self.assertTrue(ep.reward<0 or ep.reward==0 or ep.reward>.7)

    def test_prefix_with_deliveries_only_role_observation(self):
        scenario=Scenario("meeting",True,False,11)
        spec=RestartSpec(0,scenario,("PRIVATE_BOOKING","ACK_PRIVATE"),
             make_fork(scenario,("PRIVATE_BOOKING","ACK_PRIVATE")).state_hash,"HR","hr")
        ep=execute_restart_episode(spec,UniformPolicy(),0,1019)
        self.assertTrue(ep.decisions)
        self.assertEqual(ep.decisions[0].actor,"HR")
        self.assertIn("authorized_receipts",ep.decisions[0].prompt)
        for hidden in ('"success":','"violation":','"related":','"expected":'):
            self.assertNotIn(hidden,ep.decisions[0].prompt)

    def test_restarted_group_has_shared_prefix_identity(self):
        group=sample_restart_group(self.spec,UniformPolicy(),8,1717)
        self.assertEqual(len(group),8)
        self.assertEqual(len({e.initial_state_hash for e in group}),1)
        self.assertEqual(len({e.policy_id for e in group}),1)
        self.assertIsNone(group_advantages(group)) if len({e.reward for e in group})==1 else self.assertIsNotNone(group_advantages(group))
        self.assertEqual(group,[*sample_restart_group(self.spec,UniformPolicy(),8,1717)])

    def test_forged_state_hash_and_role_rejected(self):
        with self.assertRaises(RuntimeError):
            restore(replace(self.spec,state_hash="faked"))
        with self.assertRaises(RuntimeError):
            restore(replace(self.spec,role="Worker"))

    def test_policy_cannot_invent_action(self):
        with self.assertRaises(RuntimeError):
            execute_restart_episode(self.spec,BrokenPolicy(),0,23)

    def test_cce_actually_credit_first_model_action_with_valid_policy_returns(self):
        policy=FirstPolicy()
        group=sample_restart_group(self.spec,policy,8,1124)
        cce=StateDirectedPCCE(n_continuations=4,minimum_gap=.25,
                             minimum_consensus=.6)
        credits,proofs,stats=cce.make_credits(self.spec,group,policy,group_index=0)
        self.assertEqual(stats["rollouts"],8)
        self.assertEqual(stats["queried_prefixes"],1)
        self.assertEqual(stats["policy_branch_rollouts"],8)
        self.assertEqual(len(credits),8)
        self.assertEqual(stats["credited_decisions"],len(proofs))
        for proof in proofs:
            self.assertEqual(proof["decision_index"],0)
            self.assertEqual(proof["restart_state_hash"],self.spec.state_hash)
            self.assertFalse(proof["native_Qstar_used"])
            self.assertEqual({x["qstar"] for x in proof["Qpi_evidence"]["actions"].values()},{None})

    def test_robust_pair_ignores_tiny_qpi_cost_gaps(self):
        from kci.agent_rl.policy_continuation import frozen_policy_counterfactual
        r=frozen_policy_counterfactual(self.spec.to_fork(),UniformPolicy(),
                                      n_continuations=4,compute_oracle=False)
        gate=robust_pair_signal(r,n=4,minimum_gap=2.)
        self.assertFalse(gate["supported"])

    def test_cces_on_stale_policy_group_fail_closed(self):
        policy=UniformPolicy()
        group=sample_restart_group(self.spec,policy,2,211)
        group[0].policy_id="changed"
        with self.assertRaises(RuntimeError):
            StateDirectedPCCE().make_credits(self.spec,group,policy,group_index=0)

if __name__=="__main__":unittest.main()
