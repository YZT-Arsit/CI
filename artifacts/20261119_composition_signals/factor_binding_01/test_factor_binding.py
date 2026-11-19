"""Data/label/pre-registered gates before model execution: no model call."""
import unittest,json,sys
from collections import defaultdict,Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path[:0]=[str(BASE),str(HERE)]
from prepare import relation_query,sha
from build_signals import load_rows
class FactorBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads((HERE/"PLAN.json").read_text(encoding="utf-8"))
        cls.train=load_rows(HERE/"train.jsonl")
        cls.dev=load_rows(HERE/"dev.jsonl")
        cls.confirm=load_rows(HERE/"confirm_depth4.jsonl")
    def test_frozen_inputs_no_label_leakage(self):
        for name,expected in self.plan["data_sha256"].items():
            self.assertEqual(sha(HERE/(name+".jsonl")),expected)
        for x in self.train+self.dev+self.confirm:
            self.assertNotIn("safe_completion",x["task_query"])
            self.assertNotIn("safe_completion",x["relation_query"])
            self.assertNotEqual(x["task_query"],x["relation_query"])
            self.assertEqual(relation_query(x["task_query"]),x["relation_query"])
    def test_train_dev_confirm_disjoint_roots(self):
        a,b,c=[{x["base_group_id"] for x in r} for r in (self.train,self.dev,self.confirm)]
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual((len(self.train),len(self.dev),len(self.confirm)),(128,128,192))
    def test_rel_targets_really_different_from_utility(self):
        all_rows=self.train+self.dev
        values={(x["relation_target"],x["task_target"]) for x in all_rows}
        self.assertTrue({(0,0),(0,1),(1,0),(1,1)}<=values)
        self.assertEqual(Counter(x["relation_target"] for x in self.train),{0:80,1:48})
    def test_flip_and_hold_real_contrasts(self):
        for data in (self.train,self.dev,self.confirm):
            pairs=defaultdict(list)
            for x in data:
                key=(x["base_group_id"],x["role"],x["action"],x["id"].split(":")[-2])
                pairs[key].append(x)
            self.assertTrue(pairs)
            self.assertTrue(all(len(v)==2 for v in pairs.values()))
            flips=sum(v[0]["task_target"]!=v[1]["task_target"] for v in pairs.values())
            self.assertGreater(flips,0)
    def test_exact_same_schedule_both_arms(self):
        s=load_rows(HERE/"schedule.jsonl")
        trainids={x["id"] for x in self.train}
        count=Counter(k for a in s for k in a["query_ids"])
        self.assertEqual(set(count),trainids)
        self.assertEqual(set(count.values()),{2})
        self.assertTrue(all(a["actor_source_step"]==a["update"]%32 for a in s))
    def test_only_one_precommitted_question(self):
        self.assertEqual(self.plan["arms"],["task_replay_control","relation_factor"])
        self.assertEqual(self.plan["updates_per_arm"],64)
        self.assertTrue(self.plan["stop_if_fails"])
        self.assertFalse(self.plan["train_ready_for_acl"])
if __name__=="__main__":unittest.main()
