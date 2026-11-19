"""Regression tests for the bounded KCI composition dataset; no GPU/API."""
import json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build_signals,shortcut_audit,make_curriculum

class CompositionDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.out=Path(cls.temp.name)/"data"
        cls.report=build_signals.build(cls.out,2)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_native_replay_and_manifest(self):
        self.assertEqual(build_signals.audit(self.out)["checks"],"PASS")
        self.assertEqual(self.report["queries"],1536)
        self.assertEqual(self.report["pairs"],768)
        self.assertFalse(self.report["training_ready"])

    def test_stratified_split_and_flip_exposure(self):
        p=build_signals.load_rows(self.out/"pairs.jsonl")
        self.assertEqual({x["split"] for x in p},{"train","dev"})
        self.assertEqual({(x["split"],x["role"]) for x in p if x["relation_flip"]},
            {(s,r) for s in ("train","dev") for r in ("Planner","Source","Worker")})
        splits={}
        for x in p:
            splits.setdefault(x["base_group_id"],set()).add(x["split"])
        self.assertTrue(all(len(v)==1 for v in splits.values()))
        self.assertEqual(len(splits),64)

    def test_prompts_cannot_read_oracle(self):
        for q in build_signals.load_rows(self.out/"queries.jsonl"):
            self.assertNotIn("safe_completion",q["prompt"])
            self.assertNotIn("gold label",q["prompt"].lower())
            self.assertNotIn(q["world_id"],q["prompt"])
            self.assertIn("Reply ANSWER:",q["prompt"])

    def test_shortcut_reports_pair_failure(self):
        r=shortcut_audit.score(self.out)
        self.assertGreater(r["dev_accuracy"],.90)
        self.assertEqual(r["dev_flip_pair_both_correct"],0)
        self.assertEqual(r["dev_flip_pairs"],32)

    def test_curriculum_balances_pair_types_and_no_dev(self):
        make_curriculum.make(self.out)
        c=json.loads((self.out/"curriculum_report.json").read_text(encoding="utf-8"))
        self.assertEqual((c["flip_pairs"],c["hold_pairs"]),(32,32))
        self.assertEqual((c["positive"],c["negative"]),(64,64))
        q=build_signals.load_rows(self.out/"balanced_train_candidate.jsonl")
        self.assertEqual(len(q),128)
        self.assertTrue(all(x["split"]=="train" and not x["training_ready"] for x in q))
        self.assertEqual(len({x["id"] for x in q}),len(q))

    def test_tampering_rejected(self):
        p=self.out/"queries.jsonl"
        original=p.read_bytes()
        try:
            p.write_bytes(original+b" ")
            with self.assertRaises(AssertionError):build_signals.audit(self.out)
        finally:p.write_bytes(original)

    def test_no_unapproved_single_repeat(self):
        with self.assertRaises(ValueError):
            build_signals.build(Path(self.temp.name)/"invalid",1)

if __name__=="__main__":unittest.main()
