"""Causal semantic regression for repaired Horn propagation.

Tests ONLY actor-visible source data / public inference clauses; the private
environment closure is used in tests as an independent truth oracle, never
as a model input or a training target. Must hold across train, dev, test
knowledge rule families and both A-originated history variations.
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import sys,unittest
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from chain_shared_actor_cpu import SharedHornActor as OldActor,actor_feature as old_feature,stack as stack_old
from chain_shared_actor_cpu_fixed import SharedHornActor as FixedActor,actor_feature,stack
from compare_kci_chain_methods_cpu import GROUP
from kci.diverse_ci_branch_env import closure,FINAL,SLOT_LABELS
from chain_shared_actor_cpu import environment

class CausalRuleRegression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.old=OldActor()
        self.fixed=FixedActor()
    def test_same_4k_parameter_arch_and_inputs(self):
        self.assertEqual(
            sum(p.numel() for p in self.old.parameters()),
            sum(p.numel() for p in self.fixed.parameters()))
        self.fixed.load_state_dict(self.old.state_dict())
        for w in GROUP["train"][:1]:
            env=environment(w)
            st,_=env.transition(env.initial(),"A_READ_VERIFIED_PACKAGES")
            obs=env.observation(st)
            for a in ("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"):
                orig=old_feature(obs,a);fixed=actor_feature(obs,a)
                self.assertEqual(set(orig),set(fixed))
                for key in orig:
                    np.testing.assert_array_equal(orig[key],fixed[key])

    def test_exact_native_rule_closure_future_on_both_actual_A_branches(self):
        metrics={}
        failures=0
        for split,worlds in GROUP.items():
            for world in worlds:
                env=environment(world)
                initial=env.initial()
                read,_=env.transition(initial,"A_READ_VERIFIED_PACKAGES")
                for package in ("PACKAGE_A","PACKAGE_B"):
                    sent,a=env.transition(read,"A_SEND_"+package)
                    self.assertEqual(a["verdict"],"VALID")
                    self.assertTrue(sent.native.ledger)
                    state=sent
                    for action in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                        state,a=env.transition(state,action)
                        self.assertEqual(a["verdict"],"VALID")
                    obs=env.observation(state)
                    x=stack([actor_feature(obs,"OPEN_PRIVATE_CHANNEL")])
                    with torch.no_grad():
                        output=self.fixed.infer(
                            torch.maximum(x["known"],x["req"]),x["prem"],x["head"])
                        bad=self.old.infer(
                            torch.maximum(x["known"],x["req"]),x["prem"],x["head"])
                    accepted=set(x.split("::")[-1] for x in state.native.delivered["R"]
                                 if x.startswith(env.native.case_subject+"::"))
                    true=FINAL in closure(accepted|set(env.native.required_facts),
                                          env.native.graph)
                    if true: self.assertAlmostEqual(float(output.item()),1.,places=6)
                    else:
                        self.assertAlmostEqual(float(output.item()),0.,places=6)
                        failures+=int(float(bad.item())>0.5)
                    metrics.setdefault(split,0)
                    metrics[split]+=1
        self.assertEqual(metrics,{"train":32,"dev":8,"exploratory_test":16})
        self.assertGreater(failures,0)

    def test_all_six_fact_boolean_masks_exact_under_every_graph(self):
        for worlds in GROUP.values():
            for world in worlds[::4]:
                env=environment(world)
                st,_=env.transition(env.initial(),"A_READ_VERIFIED_PACKAGES")
                st,_=env.transition(st,"A_SEND_PACKAGE_A")
                st,_=env.transition(st,"INSPECT_BUSINESS")
                st,_=env.transition(st,"CHOOSE:"+env.native._route)
                obs=env.observation(st)
                base=stack([actor_feature(obs,"OPEN_PRIVATE_CHANNEL")])
                for mask in range(1<<6):
                    x={k:v.clone() for k,v in base.items()}
                    facts={chr(ord("a")+i) for i in range(6) if mask & (1<<i)}
                    for i in range(6):
                        x["known"][0,i]=1. if mask & (1<<i) else 0.
                    with torch.no_grad():
                        out=self.fixed.infer(x["known"],x["prem"],x["head"]).item()
                    target=1. if FINAL in closure(facts,env.native.graph) else 0.
                    self.assertAlmostEqual(out,target,places=6,
                                           msg=(world["graph"],mask,out,target))
if __name__=="__main__":unittest.main()
