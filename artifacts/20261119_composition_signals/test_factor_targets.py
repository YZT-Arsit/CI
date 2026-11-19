"""Read-only factor-label regression tests for KCI synthetic native tasks."""
import hashlib,json,unittest
from collections import Counter,defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
S=HERE/"pilot_01"
def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
class FactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f=rows(S/"factor_targets.jsonl")
        cls.q={x["id"]:x for x in rows(S/"queries.jsonl")}
        cls.p=rows(S/"pairs.jsonl")
        cls.quality=json.loads((S/"factor_quality.json").read_text())
    def test_complete_and_provenance(self):
        self.assertEqual(len(self.f),1536)
        self.assertEqual(len({x["id"] for x in self.f}),1536)
        self.assertEqual(self.quality["data_sha256"],sha(S/"factor_targets.jsonl"))
        self.assertTrue(all(x["training_ready"] is False for x in self.f))
    def test_labels_and_roles_unchanged(self):
        for row in self.f:
            q=self.q[row["id"]]
            self.assertEqual(q["label"],row["safe_completion"])
            self.assertEqual(q["role"],row["role"])
            self.assertEqual(q["base_group_id"],row["base_group_id"])
            self.assertEqual(q["split"],row["split"])
    def test_all_factor_grid_and_positive_relation(self):
        all_cells={(x["split"],x["role"],x["permission"],x["full_result_required"]) for x in self.f}
        self.assertEqual(len(all_cells),2*3*2*2)
        self.assertEqual(sum(x["relation_new_protected_association"] for x in self.f),384)
    def test_flip_needs_actual_relation_difference(self):
        factor={r["id"]:r for r in self.f}
        changed=0
        for p in self.p:
            if not p["relation_flip"]:continue
            x,y=factor[p["left_id"]],factor[p["right_id"]]
            self.assertNotEqual(x["relation_new_protected_association"],
                                y["relation_new_protected_association"])
            self.assertEqual(x["permission"],y["permission"])
            self.assertEqual(x["full_result_required"],y["full_result_required"])
            self.assertEqual(x["role"],y["role"])
            changed+=1
        self.assertEqual(changed,64)
if __name__=="__main__":unittest.main()
