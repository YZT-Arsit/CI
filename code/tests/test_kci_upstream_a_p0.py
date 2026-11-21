"""Regression: A public task-cost choice, causal source and strict ties.

Uses model-visible A source documents and independently saved posthoc
evaluation for assertions. Does not produce training labels or change V3.1.
"""
import sys,unittest,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R/"code"),str(R/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import environment
from kci.upstream_downstream_chain_pilot_public_contract import ChainPilotEnvPublicContract
from kci.diverse_ci_branch_env import digest

P=R/"artifacts"/"20261120_v31"
def get(path):
    return json.loads((P/path).read_text(encoding="utf-8"))

class UpstreamP0(unittest.TestCase):
    def test_original_dag_splits_hold_and_source_unchanged(self):
        root={s:{x["graph"] for x in GROUP[s]} for s in GROUP}
        self.assertEqual([len(root[x]) for x in ("train","dev","exploratory_test")],[4,1,2])
        self.assertTrue(root["train"].isdisjoint(root["dev"]|root["exploratory_test"]))
        self.assertTrue(root["dev"].isdisjoint(root["exploratory_test"]))
    def test_real_upstream_two_actions_safe_and_original_state_not_preset(self):
        for split in GROUP:
            for world in GROUP[split]:
                original=environment(world)
                enriched=ChainPilotEnvPublicContract(
                    original.spec,original.swap,original.source_mode,
                    original.safe_surcharge)
                start=enriched.initial()
                self.assertEqual(start.native.delivered["R"],[])
                state,rep=enriched.transition(start,"A_READ_VERIFIED_PACKAGES")
                self.assertEqual(rep["verdict"],"VALID")
                obs=enriched.observation(state)
                for k in ("true_Q","oracle","correct_action","value"):
                    self.assertNotIn(k,obs)
                self.assertTrue(obs["privacy_resource_contract"])
                self.assertIn("A_SEND_PACKAGE_A",obs["legal_actions"])
                self.assertIn("A_SEND_PACKAGE_B",obs["legal_actions"])
                for packet in ("PACKAGE_A","PACKAGE_B"):
                    chosen,a=enriched.transition(state,"A_SEND_"+packet)
                    self.assertEqual(a["verdict"],"VALID")
                    self.assertFalse(chosen.native.violation)
                    self.assertTrue(chosen.native.delivered["R"])
                    self.assertTrue(all(x["sender"]=="RecordsAgent"
                                        for x in chosen.native.ledger))
    def test_planner_numeric_Q_diff_exact_and_tie_count(self):
        rows=get(Path("A_PUBLIC_ONLY_COST_PLANNER_P0_01")/"RESULT.json")["predictions"]
        self.assertEqual(len(rows),28)
        self.assertTrue(all(x["exact_Qgap"] and x["correct"] for x in rows))
        self.assertEqual(len([x for x in rows if x["split"]=="exploratory_test"]),8)
        self.assertEqual(sum(x["oracle_Qgap_EVAL_ONLY"]==0 for x in rows
                             if x["split"]=="exploratory_test"),2)
        for row in rows:
            self.assertAlmostEqual(row["predicted_cost_Qgap"],
                                   row["oracle_Qgap_EVAL_ONLY"],places=7)
    def test_A_training_label_versus_Qgap_five_seed_strict_result(self):
        report=get(Path("A_PUBLIC_COST_SFT_CCE_PAIRED_P0_01")/"P0_A_SFT_VS_QSTAR.json")
        self.assertEqual(report["train_strict_action_states"],14)
        self.assertEqual(report["test_strict_states"],6)
        self.assertEqual(report["test_ties"],2)
        base=report["aggregate"]["True__optimal_action_SFT"]
        cce=report["aggregate"]["True__Qgap_reference_pair_DPO_proxy"]
        weighted=report["aggregate"]["True__Qgap_weighted_SFT"]
        self.assertEqual(base["test_strict_correct_per_seed"],[6]*5)
        self.assertEqual(cce["test_strict_correct_per_seed"],[6]*5)
        self.assertEqual(weighted["test_strict_correct_per_seed"],[4]*5)
        self.assertTrue(all(r==0 for r in base["test_mean_regret_per_seed"]))
if __name__=="__main__":unittest.main()
