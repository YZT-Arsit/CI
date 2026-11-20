"""CPU shared-agent method comparison: source-only observations and branch validity."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import sys,unittest
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from chain_shared_actor_cpu import (
    environment,actor_feature,SharedHornActor,score_actions,
    valid_decision_candidates,run_episode,digest,ROLES,
)
from compare_kci_chain_methods_cpu import (
    GROUP,METHODS,SEEDS,build_expert_dataset,
    feature_cache,supervised_loss,ref_diff,
)
from solve_v31_longterm import native_best

class SharedModelTests(unittest.TestCase):
    def test_knowledge_graph_holdout_split(self):
        graphs={k:{x["graph"] for x in GROUP[k]} for k in GROUP}
        self.assertEqual([len(GROUP[k]) for k in ("train","dev","exploratory_test")],[16,4,8])
        self.assertEqual([len(graphs[k]) for k in ("train","dev","exploratory_test")],[4,1,2])
        self.assertFalse(graphs["train"]&graphs["dev"])
        self.assertFalse(graphs["train"]&graphs["exploratory_test"])
        self.assertFalse(graphs["dev"]&graphs["exploratory_test"])

    def test_actual_A_source_data_and_action_only(self):
        env=environment(GROUP["train"][0])
        st=env.initial()
        st,a=env.transition(st,"A_READ_VERIFIED_PACKAGES")
        self.assertEqual(a["verdict"],"VALID")
        obs=env.observation(st)
        opts=valid_decision_candidates(obs,obs["legal_actions"])
        self.assertIn("A_SEND_PACKAGE_A",opts)
        self.assertIn("A_SEND_PACKAGE_B",opts)
        f1=actor_feature(obs,"A_SEND_PACKAGE_A")
        f2=actor_feature(obs,"A_SEND_PACKAGE_B")
        self.assertNotEqual(tuple(f1["bundle"]),tuple(f2["bundle"]))
        self.assertEqual(float(f1["known"].sum()),0.)
        self.assertIn("RecordsAgent",ROLES)
        for key in ("Qstar","oracle","preferred_action","true_label","reward"):
            self.assertNotIn(key,obs)

    def test_B_observation_comes_from_actual_received_source(self):
        env=environment(GROUP["train"][0])
        st=env.initial();st,_=env.transition(st,"A_READ_VERIFIED_PACKAGES")
        a=st
        st,_=env.transition(a,"A_SEND_PACKAGE_A")
        for action in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
            st,x=env.transition(st,action)
            self.assertEqual(x["verdict"],"VALID")
        obs=env.observation(st)
        self.assertEqual(obs["actor"],"Coordinator")
        self.assertGreater(len(obs["delivered_message_log"]),0)
        before=actor_feature(obs,"OPEN_PRIVATE_CHANNEL")
        self.assertGreater(float(before["known"].sum()),0)
        self.assertEqual(float(before["src"].sum()),0)
        self.assertIn("OPEN_PRIVATE_CHANNEL",obs["legal_actions"])
        self.assertIn("SKIP_PRIVATE_CHANNEL",obs["legal_actions"])

    def test_true_same_state_B_Qgap_switch(self):
        row=GROUP["train"][0]
        env=environment(row)
        first=env.initial();first,_=env.transition(first,"A_READ_VERIFIED_PACKAGES")
        world={}
        for pkg in ("PACKAGE_A","PACKAGE_B"):
            s,_=env.transition(first,"A_SEND_"+pkg)
            for a in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                s,_=env.transition(s,a)
            openq=native_best(env.native,s.native,"OPEN_PRIVATE_CHANNEL")
            skipq=native_best(env.native,s.native,"SKIP_PRIVATE_CHANNEL")
            world[pkg]=(env.observation(s),openq["native_exact_Qstar"]-skipq["native_exact_Qstar"])
        a=world[row["related_package"]]
        b=world[row["benign_package"]]
        self.assertGreater(a[1],0)
        self.assertLess(b[1],0)
        obsA=a[0].copy();obsB=b[0].copy()
        ha=obsA.pop("delivered_message_log")
        hb=obsB.pop("delivered_message_log")
        self.assertEqual(obsA,obsB)
        self.assertNotEqual(ha,hb)

    def test_train_expert_CCE_actions_are_always_authorized(self):
        routine,cf,pair,counts=build_expert_dataset(GROUP["train"][:2])
        self.assertGreater(len(routine),0)
        self.assertGreater(len(cf),len(routine))
        self.assertGreater(len(pair),0)
        self.assertTrue(any(x["actor"]=="BusinessExecutor" for x in routine))
        self.assertTrue(any(x["actor"]=="Coordinator" for x in routine))
        self.assertTrue(any(x["actor"]=="RecordsAgent" for x in routine))
        self.assertTrue(all(x["chosen"] in x["options"] for x in routine+cf+pair))
        self.assertTrue(all(x["rejected"] in x["options"] for x in pair))
        self.assertTrue(all(x["rejected"]!=x["chosen"] for x in pair))

    def test_shared_actor_same_weight_set_for_A_B_C(self):
        env=environment(GROUP["train"][0])
        st=env.initial();st,_=env.transition(st,"A_READ_VERIFIED_PACKAGES")
        obs=env.observation(st)
        model=SharedHornActor()
        param_ids={id(p) for p in model.parameters()}
        self.assertGreater(sum(p.numel() for p in model.parameters()),1000)
        logits=score_actions(model,obs,["A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"])
        self.assertEqual(list(logits.shape),[2])
        self.assertEqual(param_ids,{id(p) for p in model.parameters()})
        self.assertEqual(METHODS[0],"action_sft")

    def test_actual_policy_emits_valid_logical_native_transitions(self):
        torch.manual_seed(7)
        env=environment(GROUP["train"][0])
        model=SharedHornActor()
        ep=run_episode(env,model,greedy=True)
        self.assertFalse(ep["invalid"])
        self.assertTrue(ep["done"])
        self.assertEqual(ep["n_policy_choices"],len(ep["decision_states"]))
        for item in ep["decision_states"]:
            self.assertIn(item["selected_action"],item["candidate_actions"])
            self.assertEqual(item["selected_action"] in item["actor_observation"]["legal_actions"],True)

if __name__=="__main__":unittest.main()
