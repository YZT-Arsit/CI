"""Executable KCI receiver-knowledge seed v2 tests, all CPU-only."""
import unittest,sys
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.knowledge_world_seed_v2 import (
    build_seeds,KnowledgeTask,certified_contrasts,topological_orders,FACTS
)

class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.seeds=build_seeds()
    def test_20_diverse_structures_and_held_out_topology(self):
        self.assertEqual(len(self.seeds),20)
        self.assertEqual(len({s.family for s in self.seeds}),20)
        self.assertEqual({s.topology for s in self.seeds if s.split=="test"},{"conjunction"})
        self.assertEqual(len({s.anchor for s in self.seeds}),4)
    def test_all_real_flip_hold_permission_and_order(self):
        changes=0
        for seed in self.seeds:
            with self.subTest(seed=seed.seed_id):
                p=certified_contrasts(seed)
                a=p["probe"]
                self.assertEqual(a["linked"]["action"],"SEND_RAW_"+seed.anchor)
                self.assertEqual(a["linked"]["audit"]["verdict"],"VIOLATION")
                self.assertEqual(a["unrelated"]["audit"]["verdict"],"VALID")
                self.assertEqual(a["hold_a"]["audit"]["verdict"],a["hold_b"]["audit"]["verdict"])
                self.assertEqual(a["linked_authorized"]["audit"]["verdict"],"VALID")
                self.assertTrue(p["safe_witness"]["success"])
                self.assertTrue(p["unsafe_raw"]["violation"])
                self.assertFalse(KnowledgeTask(seed,"linked",False,"full").rollout(
                    topological_orders(seed,FACTS)[0],FACTS)["success"])
                changes+=int(p["order_has_distinct_trigger_step"])
        self.assertEqual(changes,15)
    def test_action_observation_contains_no_evaluator_truth(self):
        for seed in self.seeds:
            for history in ("linked","unrelated","hold_a","hold_b","empty"):
                env=KnowledgeTask(seed,history,False)
                st=env.initial()
                obs=env.observation(st)
                for field in ("history","violation","reward","gold_action",
                              "seed_id","secret_derived","test_label","knowledge_truth"):
                    self.assertNotIn(field,obs)
                self.assertIn("authorized_delivered_receipts",obs)
                self.assertEqual(set(env.options(st)),set(obs["available_actions"]))
    def test_no_full_task_without_delivered_work(self):
        for seed in self.seeds:
            env=KnowledgeTask(seed,"linked",False,"full")
            s=env.initial()
            s,receipt=env.transition(s,"ABORT")
            self.assertTrue(s.done)
            self.assertFalse(s.success)
            self.assertFalse(s.violation)

if __name__=="__main__":unittest.main()
