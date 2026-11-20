"""CPU tests for new executable receiver-knowledge seed tasks."""
from pathlib import Path
import sys
import unittest
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.knowledge_world_seed import (
    DOMAINS,RULES,Seed,KnowledgeTask,build_seeds,certified_contrasts,
    topological_orders,find_safe_witness,FACTS,digest,
)

class KnowledgeSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seeds=build_seeds()
    def test_structural_coverage_and_disjoint_splits(self):
        self.assertEqual(len(self.seeds),20)
        self.assertEqual(len({(s.domain,s.topology) for s in self.seeds}),20)
        self.assertEqual(len({s.seed_id for s in self.seeds}),20)
        self.assertEqual({s.topology for s in self.seeds if s.split=="test"},{"conjunction"})
        self.assertEqual({s.domain for s in self.seeds if s.split=="test"},set(DOMAINS))
        self.assertTrue({s.split for s in self.seeds}=={"train","dev","test"})
        self.assertEqual(len(RULES),4)
    def test_all_twenty_real_flip_hold_order_path_contrasts(self):
        for seed in self.seeds:
            with self.subTest(seed=seed.seed_id):
                proof=certified_contrasts(seed)
                self.assertEqual(proof["probe"]["linked"]["action"],"SEND_RAW_f0")
                self.assertTrue(proof["flip_pair_passed"] and proof["hold_pair_passed"])
                self.assertTrue(proof["safe_witness"]["success"])
                self.assertFalse(proof["safe_witness"]["violation"])
                self.assertTrue(proof["unsafe_raw"]["violation"])
                self.assertTrue(proof["order_first"]["first_derived"])
                self.assertTrue(proof["order_last"]["first_derived"])
    def test_no_single_raw_clue_leaks_without_history(self):
        for seed in self.seeds:
            for f in FACTS:
                env=KnowledgeTask(seed,"empty",False,profile="full")
                s=env.initial()
                # Evaluate a legitimate isolated delivery with other business
                # prerequisites satisfied by attestation, not raw exposure.
                for dep in FACTS:
                    if dep==f:continue
                    if "PROCESS_"+dep in env.options(s):
                        s,_=env.transition(s,"PROCESS_"+dep)
                        s,_=env.transition(s,"VERIFY_PRIVATE_"+dep)
                        s,_=env.transition(s,"SEND_ATTEST_"+dep)
                if "PROCESS_"+f not in env.options(s):
                    continue
                s,_=env.transition(s,"PROCESS_"+f)
                s,a=env.transition(s,"SEND_RAW_"+f)
                self.assertNotEqual(a["verdict"],"VIOLATION")
    def test_no_hidden_eval_field_in_actual_actor_observation(self):
        seed=self.seeds[0]
        for history in ("linked","unrelated","hold_a","hold_b","empty"):
            env=KnowledgeTask(seed,history,False)
            st=env.initial()
            obs=env.observation(st)
            self.assertEqual(obs["role"],env.actor(st))
            for bad in ("history","authorized","secret_derived","expected_action",
                        "correct_action","success","violation","reward","split"):
                self.assertNotIn(bad,obs)
            if history!="empty":
                self.assertTrue(obs["authorized_delivered_receipts"])
            else:
                self.assertEqual(obs["authorized_delivered_receipts"],[])
    def test_all_private_and_refuse_dont_solve_full_task(self):
        for seed in self.seeds:
            env=KnowledgeTask(seed,"linked",False,"full")
            order=topological_orders(seed,env.required)[0]
            allpriv=env.rollout(order,env.required)
            self.assertFalse(allpriv["success"])
            self.assertFalse(allpriv["violation"])
            s=env.initial()
            s,_=env.transition(s,"ABORT")
            self.assertFalse(s.success)
    def test_history_knowledge_is_not_marked_as_current_utility(self):
        for seed in self.seeds:
            env=KnowledgeTask(seed,"linked",False,"minimal")
            s=env.initial()
            self.assertFalse(s.success)
            self.assertFalse(s.completed)
            self.assertNotIn("PRIVATE",env.infer(s))
            self.assertNotIn("DELIVER_RESULT",env.options(s))
    def test_unauthorized_and_authorized_same_knowledge_update(self):
        for seed in self.seeds:
            a=KnowledgeTask(seed,"linked",False,"minimal")
            b=KnowledgeTask(seed,"linked",True,"minimal")
            sa=a.initial();sb=b.initial()
            for env in (a,b):
                self.assertIn("PROCESS_f0",env.options(env.initial()))
            sa,_=a.transition(sa,"PROCESS_f0")
            sb,_=b.transition(sb,"PROCESS_f0")
            sa,va=a.transition(sa,"SEND_RAW_f0")
            sb,vb=b.transition(sb,"SEND_RAW_f0")
            self.assertEqual(va["supporting_raw_facts"],vb["supporting_raw_facts"])
            self.assertEqual(va["verdict"],"VIOLATION")
            self.assertEqual(vb["verdict"],"VALID")
            self.assertTrue(sa.violation)
            self.assertFalse(sb.violation)

if __name__=="__main__":unittest.main()
