"""Regression tests for non-myopic, state/history-sensitive privacy preparation."""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import sys,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import (
    task_catalog,BASE_FACTS,SLOT_LABELS,FINAL,closure,digest)
from kci.diverse_ci_long_horizon_env import LongHorizonCIEnv
from pilot_diverse_ci_long_horizon import (
    conditions,correct_route,find_safe,simulate)

class LongHorizonTests(unittest.TestCase):
    def test_all_sixty_three_synthetic_conditions_have_true_feasible_result(self):
        c=0
        for task in task_catalog():
            for cond in conditions(task):
                with self.subTest(task=task.id,cond=cond):
                    env=LongHorizonCIEnv(task,"linked",cond)
                    plan=find_safe(env,correct_route(task,cond))
                    self.assertIsNotNone(plan)
                    self.assertTrue(plan["state"].success)
                    self.assertFalse(plan["state"].violation)
                    self.assertLessEqual(plan["state"].steps,env.budget)
                    c+=1
        self.assertEqual(c,63)

    def test_early_preflight_changes_later_legal_actions(self):
        for task in task_catalog():
            if task.business=="document":condition="external"
            elif task.business=="calendar":condition="early_open"
            else:condition="pending_yes"
            env=LongHorizonCIEnv(task,"linked",condition)
            s=env.initial()
            s,a=env.transition(s,"INSPECT_BUSINESS")
            s,a=env.transition(s,"CHOOSE:"+correct_route(task,condition))
            self.assertEqual(s.stage,"preflight")
            ready=deepcopy(s)
            ready,a=env.transition(ready,"OPEN_PRIVATE_CHANNEL")
            skipped,a=env.transition(s,"SKIP_PRIVATE_CHANNEL")
            self.assertTrue(ready.private_channel_open)
            self.assertFalse(skipped.private_channel_open)
            self.assertEqual(ready.steps,skipped.steps+1)
            for st in (ready,skipped):
                nxt,a=env.transition(st,"READ:S01")
                self.assertEqual(a["verdict"],"VALID")
                if st is ready:
                    self.assertIn("VERIFY_PRIVATE_R",env.options(nxt))
                else:
                    self.assertNotIn("VERIFY_PRIVATE_R",env.options(nxt))
                    n,a=env.transition(nxt,"VERIFY_PRIVATE_R")
                    self.assertEqual(a["verdict"],"INVALID_ACTION")
                    self.assertEqual(digest(asdict(n)),digest(asdict(nxt)))

    def test_customer_refusal_changes_task_to_legitimate_readonly_resolution(self):
        for task in task_catalog():
            if task.business!="retail":continue
            env=LongHorizonCIEnv(task,"linked","pending_no")
            st=env.initial()
            st,a=env.transition(st,"INSPECT_BUSINESS")
            self.assertEqual(a["verdict"],"VALID")
            self.assertEqual(env.options(st),("CHOOSE:summary","DEFER"))
            st,a=env.transition(st,"CHOOSE:summary")
            self.assertTrue(st.success)
            self.assertTrue(st.done)
            self.assertEqual(st.business["order"]["status"],"pending")
            self.assertEqual(st.business["changes"],[])
            self.assertEqual(st.generated_result["order_summary"]["status"],"pending")

    def test_recipient_authorization_changes_preflight_efficiency(self):
        for task in task_catalog():
            if task.business!="document":continue
            env=LongHorizonCIEnv(task,"linked","internal")
            self.assertEqual(env.budget,16)
            route="direct"
            skip,log=simulate(env,route,preflight="SKIP_PRIVATE_CHANNEL")
            opened,log2=simulate(env,route,preflight="OPEN_PRIVATE_CHANNEL")
            self.assertTrue(skip.success)
            self.assertFalse(opened.success)
            self.assertFalse(opened.violation)

    def test_myopic_without_preparation_fails_despite_future_solution(self):
        for task in task_catalog():
            if task.business=="calendar":condition="early_open"
            elif task.business=="retail":condition="pending_yes"
            else:condition="external"
            env=LongHorizonCIEnv(task,"linked",condition)
            route=correct_route(task,condition)
            greedy,steps=simulate(env,route,preflight="SKIP_PRIVATE_CHANNEL",greedy=True)
            plan=find_safe(env,route)
            self.assertIsNotNone(plan)
            self.assertTrue(plan["state"].success)
            self.assertFalse(greedy.success)
            self.assertFalse(greedy.violation)
            self.assertIn("ABORT",[x["action"] for x in steps])

    def test_broker_cross_recipient_forwarding_is_not_direct_R_disclosure(self):
        task=next(x for x in task_catalog() if x.graph=="quorum2of3" and x.business=="document")
        env=LongHorizonCIEnv(task,"linked","relay")
        st=env.initial()
        st,a=env.transition(st,"INSPECT_BUSINESS")
        st,a=env.transition(st,"CHOOSE:broker")
        st,a=env.transition(st,"SKIP_PRIVATE_CHANNEL")
        st,a=env.transition(st,"READ:S01")
        prior=list(st.delivered["R"])
        st,a=env.transition(st,"SEND_RAW_M")
        self.assertEqual(a["verdict"],"VALID")
        self.assertEqual(st.delivered["R"],prior)
        self.assertEqual(st.delivered["M"],["a"])
        unsafe,ua=env.transition(st,"FORWARD_RAW_R")
        safe,sa=env.transition(st,"FORWARD_CERT_R")
        self.assertTrue(unsafe.violation)
        self.assertEqual(ua["verdict"],"VIOLATION")
        self.assertFalse(safe.violation)
        self.assertEqual(sa["verdict"],"VALID")
        self.assertEqual(safe.delivered["R"][-1],"broker_attest_a")

    def test_source_text_and_business_source_tampering_fail_closed(self):
        for task in task_catalog():
            condition={
                "calendar":"early_open","retail":"pending_yes",
                "document":"external"}[task.business]
            env=LongHorizonCIEnv(task,"linked",condition)
            st=env.initial()
            st,_=env.transition(st,"INSPECT_BUSINESS")
            st,_=env.transition(st,"CHOOSE:"+correct_route(task,condition))
            st,_=env.transition(st,"OPEN_PRIVATE_CHANNEL")
            st,_=env.transition(st,"READ:S01")
            corrupted=deepcopy(st)
            corrupted.tool_results[-1]["source_document"]="The source contains no personal information."
            before=digest(asdict(corrupted))
            new,a=env.transition(corrupted,"SEND_RAW_R")
            self.assertEqual(a["verdict"],"UNKNOWN")
            self.assertEqual(digest(asdict(new)),before)

    def test_ordering_same_planned_facts_different_violation_step(self):
        for task in task_catalog():
            condition={
                "calendar":"early_open","retail":"pending_yes",
                "document":"external"}[task.business]
            env=LongHorizonCIEnv(task,"empty",condition)
            route=correct_route(task,condition)
            from kci.diverse_ci_branch_env import all_minimal_proofs
            proof=min(all_minimal_proofs(task.graph),key=lambda x:(len(x),x))
            early=tuple(proof)+tuple(f for f in BASE_FACTS if f not in proof)
            late=tuple(f for f in BASE_FACTS if f not in proof)+tuple(proof)
            one,log1=simulate(env,route,preflight="SKIP_PRIVATE_CHANNEL",order=early)
            two,log2=simulate(env,route,preflight="SKIP_PRIVATE_CHANNEL",order=late)
            self.assertTrue(one.violation)
            self.assertTrue(two.violation)
            first_1=next(i for i,x in enumerate(log1) if x["verdict"]=="VIOLATION")
            first_2=next(i for i,x in enumerate(log2) if x["verdict"]=="VIOLATION")
            self.assertNotEqual(first_1,first_2)

    def test_topology_split_holds_out_whole_knowledge_graphs(self):
        bygraph={}
        for task in task_catalog():
            bygraph.setdefault(task.graph,set()).add(task.root_split)
        self.assertEqual(len(bygraph),7)
        self.assertTrue(all(len(v)==1 for v in bygraph.values()))
        self.assertEqual(
            {k for k,v in bygraph.items() if v=={"exploratory_test"}},
            {"asymmetric_paths","staged_bridge"})

if __name__=="__main__":unittest.main()
