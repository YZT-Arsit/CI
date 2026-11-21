"""Narrow A-agent public resource-cost observation repair, no oracle leakage.

This is optional and does not change frozen V3.1/old native transitions.
It makes same task privacy setup cost available upstream, as already visible
to Coordinator, without exposing protected closure or preferred choice.
"""
from pathlib import Path
from dataclasses import asdict
import json,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.upstream_downstream_chain_pilot import ChainPilotEnv
from kci.upstream_downstream_chain_pilot_public_contract import ChainPilotEnvPublicContract
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import environment,actor_feature
from kci.diverse_ci_branch_env import digest

class AResourceContractObservation(unittest.TestCase):
    def test_public_contract_upstream_no_other_changes(self):
        for world in GROUP["train"][:2]+GROUP["exploratory_test"][:2]:
            original=environment(world)
            enriched=ChainPilotEnvPublicContract(
                original.spec,original.swap,original.source_mode,
                original.safe_surcharge)
            for action in ("A_READ_VERIFIED_PACKAGES","A_SEND_PACKAGE_A"):
                if action=="A_READ_VERIFIED_PACKAGES":
                    x=original.initial();y=enriched.initial()
                ox=original.observation(x)
                oy=enriched.observation(y)
                public=oy.pop("privacy_resource_contract")
                self.assertEqual(ox,oy)
                self.assertEqual(public,x.native.business["privacy_resource_contract"])
                self.assertTrue(set(("private_channel_setup_steps","skip_setup_steps",
                                     "task_budget","step_cost"))<=set(public))
                self.assertNotIn("optimal_action",public)
                self.assertNotIn("oracle",public)
                a=x;b=y
                x,rx=original.transition(a,action)
                y,ry=enriched.transition(b,action)
                self.assertEqual(rx["verdict"],ry["verdict"])
                self.assertEqual(digest(asdict(x.native)),digest(asdict(y.native)))
            # Encoder legitimately gains the public actual preflight fee.
            f=actor_feature(enriched.observation(enriched.transition(
                 enriched.initial(),"A_READ_VERIFIED_PACKAGES")[0]),
                 "A_SEND_PACKAGE_A")
            self.assertAlmostEqual(float(f["extra"][2]),
                enriched.spec.prep_fee/5)
            self.assertAlmostEqual(float(f["extra"][3]),
                enriched.spec.cost_per_step/0.05)

if __name__=="__main__":unittest.main()
