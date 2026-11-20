"""CPU tests for the isolated public-workflow-inspired KCI pilot."""
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import json,sys,unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci import (
    PUBLIC_TASKS,PublicCIEnv,check_goal,graph_texts,SEEDS,TOOL_ROLE,V
)
from kci.knowledge_world_seed_v2 import FACTS,topological_orders,find_safe_witness,digest

class PublicWorkflowCITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs=PUBLIC_TASKS

    def test_independent_source_anchoring_and_splitable_heterogeneity(self):
        self.assertEqual(len(self.specs),10)
        self.assertEqual(len({x.id for x in self.specs}),10)
        self.assertEqual(Counter(x.suite for x in self.specs),
                         {"agentdojo_workspace":5,"tau2_retail":5})
        self.assertEqual(len({x.upstream_id for x in self.specs if x.suite=="agentdojo_workspace"}),5)
        self.assertEqual(len({x.upstream_id for x in self.specs if x.suite=="tau2_retail"}),5)
        self.assertEqual({SEEDS[x.seed_id].topology for x in self.specs},
                         {"pair","chain","fork","conjunction"})
        self.assertTrue(all(len(x.read_tools)==4 and x.commit_tools for x in self.specs))

    def test_same_message_dynamic_knowledge_flip_hold_permission(self):
        for spec in self.specs:
            with self.subTest(root=spec.id):
                out={}
                for name in ("linked","unrelated","hold_a","hold_b","linked_authorized"):
                    env=PublicCIEnv(spec,"linked" if name=="linked_authorized" else name,
                                    name=="linked_authorized","minimal")
                    s=env.initial()
                    if "AUTHENTICATE_CUSTOMER" in env.options(s):
                        s,meta=env.transition(s,"AUTHENTICATE_CUSTOMER")
                        self.assertEqual(meta["verdict"],"VALID")
                    action=env.encode_action("PROCESS_"+env.seed.anchor)
                    self.assertIn(action,env.options(s))
                    s,_=env.transition(s,action)
                    obs=env.observation(s)
                    self.assertEqual(obs["actor"],TOOL_ROLE[spec.read_tools[int(env.seed.anchor[1:])]])
                    self.assertEqual(len(obs["available_actions"]),2)
                    self.assertTrue(any("authorized_source_document" in x["message"] for x in obs["tool_results"]))
                    after,audit=env.transition(s,env.encode_action("SEND_RAW_"+env.seed.anchor))
                    out[name]=audit
                self.assertEqual(out["linked"]["verdict"],"VIOLATION")
                self.assertEqual(out["unrelated"]["verdict"],"VALID")
                self.assertEqual(out["hold_a"]["verdict"],out["hold_b"]["verdict"])
                self.assertEqual(out["hold_a"]["verdict"],"VALID")
                self.assertTrue(out["linked_authorized"]["new_derived_private"])
                self.assertEqual(out["linked_authorized"]["verdict"],"VALID")
                self.assertEqual(out["linked"]["supporting_facts"],
                                 out["linked_authorized"]["supporting_facts"])

    def test_all_tasks_reach_real_mutated_result_with_safe_path(self):
        for spec in self.specs:
            with self.subTest(root=spec.id):
                env=PublicCIEnv(spec,"linked",False,"full")
                proof=find_safe_witness(env.knowledge)
                self.assertIsNotNone(proof)
                safe=env.run_script(proof["order"],proof["privately_verified"])
                self.assertTrue(safe["success"])
                self.assertFalse(safe["violation"])
                self.assertIn(spec.commit_tools[-1],[x["name"] for x in safe["tool_calls"]])
                self.assertEqual(len([x for x in safe["trace"] if x["action"]=="COMPLETE_BUSINESS_TASK"]),1)
                for x in safe["tool_calls"]:
                    self.assertIn("args",x)
                    self.assertTrue(isinstance(x["args"],dict))
                    self.assertNotEqual(x["args"],{"fixture_id":spec.id})

    def test_full_task_requires_selective_data_not_blanket_disclosure(self):
        for spec in self.specs:
            env=PublicCIEnv(spec,"linked",False,"full")
            order=topological_orders(env.seed,FACTS)[0]
            raw=env.run_script(order,())
            allpriv=env.run_script(order,FACTS)
            self.assertTrue(raw["violation"])
            self.assertFalse(raw["success"])
            self.assertFalse(allpriv["success"])
            self.assertFalse(allpriv["violation"])

    def test_retail_authentication_confirmation_and_order_preconditions(self):
        for spec in self.specs:
            if spec.suite!="tau2_retail":continue
            with self.subTest(root=spec.id):
                env=PublicCIEnv(spec,"unrelated",False,"full")
                st=env.initial()
                self.assertEqual(env.options(st),("AUTHENTICATE_CUSTOMER","ABORT"))
                old=st.store["customer"]["authenticated"]
                self.assertFalse(old)
                nxt,a=env.transition(st,"AUTHENTICATE_CUSTOMER")
                self.assertTrue(nxt.store["customer"]["authenticated"])
                self.assertFalse(st.store["customer"]["authenticated"])
                # No direct completion before all four work receipts.
                x,audit=env.transition(nxt,"COMPLETE_BUSINESS_TASK")
                self.assertEqual(audit["verdict"],"INVALID_ACTION")
                self.assertFalse(x.native.success)

    def test_model_view_has_no_evaluator_feedback_or_raw_fact_ids(self):
        for spec in self.specs:
            env=PublicCIEnv(spec,"linked",False,"minimal")
            s=env.initial()
            if "AUTHENTICATE_CUSTOMER" in env.options(s):
                s,_=env.transition(s,"AUTHENTICATE_CUSTOMER")
            obs=env.observation(s)
            for key in ("seed_id","history","authorized","violation","reward",
                        "supporting_facts","native","ground_truth","best_action","fact_permutation"):
                self.assertNotIn(key,obs)
            self.assertTrue(obs["request"])
            self.assertTrue(obs["fictional_privacy_policy"])
            self.assertNotIn("=>",json.dumps(obs))

    def test_order_permutation_may_change_when_breach_occurs(self):
        temporal=0;actor=0
        for spec in self.specs:
            env=PublicCIEnv(spec,"empty",False,"full")
            orders=topological_orders(env.seed,FACTS)
            proofs=[env.run_script(order,()) for order in orders]
            failures=[r for r in proofs if r["violation"]]
            self.assertTrue(failures)
            def locator(x):
                for idx,a in enumerate(x["trace"]):
                    if a.get("private_derived"):return idx,a["action"]
                raise RuntimeError("missing breach")
            sources={locator(x) for x in failures}
            self.assertGreater(len(sources),1)
            temporal+=int(len({x[0] for x in sources})>1)
            actor+=int(len({x[1] for x in sources})>1)
        self.assertGreaterEqual(temporal,5)
        self.assertGreaterEqual(actor,5)

if __name__=="__main__":unittest.main()
