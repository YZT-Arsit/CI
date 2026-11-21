"""Independently verify action-conditioned public-rule repair.

The actor receives ONLY the legal per-role observation. Exact native
transition results are used exclusively in assertions; no outcome or
oracle label is passed to the neural scorer or trained inside the test.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import sys,unittest
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.diverse_ci_branch_env import FINAL,SLOT_LABELS,closure
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import (
    SharedHornActor,actor_feature,stack,environment,ACTIONS,
)
from chain_shared_actor_cpu_fixed import SharedHornActor as OnlyFixedActor

class ActionRiskCausality(unittest.TestCase):
    def test_same_user_visible_encoder_and_zero_initialized_interaction(self):
        env=environment(GROUP["train"][0])
        st,_=env.transition(env.initial(),"A_READ_VERIFIED_PACKAGES")
        obs=env.observation(st)
        torch.manual_seed(7)
        model=SharedHornActor()
        torch.manual_seed(7)
        old=OnlyFixedActor()
        same_gain=model.gain.state_dict()
        other_gain=old.gain.state_dict()
        self.assertTrue(all(torch.allclose(v,other_gain[k]) for k,v in same_gain.items()))
        self.assertEqual(model.risk_action_head.weight.shape,(1,5))
        self.assertTrue(torch.all(model.risk_action_head.weight==0))
        for a in ("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"):
            inp=actor_feature(obs,a)
            self.assertFalse(any(k.lower().startswith(("gold","oracle","reward","qstar"))
                                 for k in inp))
            self.assertIn("prem",inp)
            self.assertEqual(inp["known"].tolist(),[0.]*6)

    def test_future_risk_from_real_A_is_contextual_not_raw_label(self):
        count=0
        for split in GROUP:
            for row in GROUP[split]:
                env=environment(row)
                st,_=env.transition(env.initial(),"A_READ_VERIFIED_PACKAGES")
                for pkg in ("PACKAGE_A","PACKAGE_B"):
                    delivered,_=env.transition(st,"A_SEND_"+pkg)
                    assert delivered.native.delivered["R"]
                    s=delivered
                    for ac in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                        s,_=env.transition(s,ac)
                    obs=env.observation(s)
                    x=stack([actor_feature(obs,a) for a in (
                        "OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")])
                    model=SharedHornActor()
                    with torch.no_grad():
                        out=model.infer(torch.maximum(x["known"],x["req"]),
                                        x["prem"],x["head"])
                    facts=set(z.split("::")[-1] for z in delivered.native.delivered["R"])
                    target=FINAL in closure(facts|set(env.native.required_facts),env.native.graph)
                    self.assertEqual(float(out[0]),float(target))
                    self.assertEqual(float(out[1]),float(target))
                    self.assertEqual(pkg==row["related_package"],target)
                    count+=1
        self.assertEqual(count,56)

    def test_raw_source_risk_is_native_incremental_violation(self):
        tested=0;violating=0;safe=0
        for w in GROUP["train"][::2]+GROUP["exploratory_test"][::2]:
            env=environment(w)
            st,_=env.transition(env.initial(),"A_READ_VERIFIED_PACKAGES")
            st,_=env.transition(st,"A_SEND_"+w["related_package"])
            st,_=env.transition(st,"INSPECT_BUSINESS")
            st,_=env.transition(st,"CHOOSE:"+env.native._route)
            st,_=env.transition(st,"OPEN_PRIVATE_CHANNEL")
            for f in env.native.required_facts:
                st,_=env.transition(st,"READ:"+SLOT_LABELS[f])
                obs=env.observation(st)
                inp=stack([actor_feature(obs,"SEND_RAW_R")])
                model=SharedHornActor()
                with torch.no_grad():
                    est=float(model.infer(
                        torch.maximum(inp["known"],inp["src"]),
                        inp["prem"],inp["head"])[0])
                _,audit=env.transition(st,"SEND_RAW_R")
                danger=audit["verdict"]=="VIOLATION"
                self.assertEqual(est,float(danger),(w["world"],f,obs.get("most_recent_tool_output")))
                tested+=1;violating+=int(danger);safe+=int(not danger)
                if danger:
                    st,actual=env.transition(st,"VERIFY_PRIVATE_R")
                    self.assertEqual(actual["verdict"],"VALID")
                    self.assertFalse(st.native.violation)
                    st,actual=env.transition(st,"SEND_CERT_R")
                    self.assertEqual(actual["verdict"],"VALID")
                else:
                    st,actual=env.transition(st,"SEND_RAW_R")
                    self.assertEqual(actual["verdict"],"VALID")
        self.assertGreater(violating,0)
        self.assertGreater(safe,0)
        self.assertGreater(tested,10)

    def test_option_specific_gradient_depends_on_current_public_future_risk(self):
        env=environment(GROUP["train"][0])
        st,_=env.transition(env.initial(),"A_READ_VERIFIED_PACKAGES")
        st,_=env.transition(st,"A_SEND_"+GROUP["train"][0]["related_package"])
        st,_=env.transition(st,"INSPECT_BUSINESS")
        st,_=env.transition(st,"CHOOSE:"+env.native._route)
        obs=env.observation(st)
        x=stack([actor_feature(obs,a) for a in (
            "OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")])
        torch.manual_seed(7)
        model=SharedHornActor()
        prob=torch.log_softmax(model(x),dim=0)
        loss=-prob[0]
        loss.backward()
        grad=model.risk_action_head.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertNotEqual(float(grad[0,1]),0.)
        self.assertNotEqual(float(grad[0,2]),0.)
        self.assertEqual(float(grad[0,0]),0.)

if __name__=="__main__":unittest.main()
