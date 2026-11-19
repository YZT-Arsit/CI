"""Read-only validation for KCI multi-step RL-signal probe."""
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json,sys,unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import rl_signal as s
def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=HERE/"pilot"
        cls.report=json.loads((cls.out/"SUMMARY.json").read_text(encoding="utf-8"))
        cls.manifest=json.loads((cls.out/"MANIFEST.json").read_text(encoding="utf-8"))
        cls.starts=load(cls.out/"starts.jsonl")
        cls.rollouts=load(cls.out/"rollouts.jsonl")
        cls.groups=load(cls.out/"groups.jsonl")
    def test_immutable_artifacts(self):
        for path,h in self.manifest.items():
            self.assertEqual(sha256((self.out/path).read_bytes()).hexdigest(),h)
        self.assertEqual(len(self.rollouts),256)
        self.assertEqual(len(self.starts),32)
    def test_risk_reward_and_completion(self):
        for x in self.rollouts:
            outcome=x["outcome"]
            if outcome["ci_violation"]:
                self.assertEqual(outcome["reward"],-3.)
                self.assertFalse(outcome["success"])
            elif outcome["success"]:
                self.assertTrue(outcome["done"])
                self.assertGreater(outcome["reward"],1.)
            else:
                self.assertEqual(outcome["reward"],-.25)
            self.assertTrue(all(a["selected_action"] in a["valid_actions"] for a in x["trajectory"]))
    def test_nonzero_advantages_informative(self):
        self.assertEqual(sum(g["informative"] for g in self.groups),29)
        for g in self.groups:
            self.assertEqual(len(g["advantages"]),8)
            self.assertLess(abs(sum(g["advantages"])),1e-4)
            self.assertEqual(g["informative"],max(g["rewards"])>min(g["rewards"]))
    def test_current_legal_but_different_futures(self):
        self.assertEqual(self.report["both_legal_actions_with_future_value_gap"],32)
        for row in self.starts:
            val=list(row["initial_state_options"].values())
            self.assertEqual(len(val),2)
            self.assertTrue(all(x["immediate_ci"]=="VALID" for x in val))
            self.assertNotEqual(tuple((x["safe_success_reachable"],x["optimal_total_steps"]) for x in val)[0],
                                tuple((x["safe_success_reachable"],x["optimal_total_steps"]) for x in val)[1])
    def test_no_label_information_in_agent_observations(self):
        for e in self.rollouts:
            for t in e["trajectory"]:
                self.assertTrue(all(k not in t["observation"] for k in ("success","violation","related","expected")))
                self.assertIn("authorized_receipts",t["observation"])
                self.assertIn("derived_relation_permission",t["observation"])
if __name__=="__main__":unittest.main()
