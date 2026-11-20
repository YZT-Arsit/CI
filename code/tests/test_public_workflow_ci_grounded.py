"""W01/R03 grounded business-and-knowledge certificate regression tests (CPU)."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json,sys,unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci import LOOKUP
from kci.public_workflow_ci_grounded import (
    GroundedPublicCIEnv, EvidenceRejected, safe_text_assertion,parse_text_assertion,
    normalized_claim,source_result,root_source_facts,VERSION,
)
from kci.knowledge_world_seed_v2 import FACTS,find_safe_witness,topological_orders,digest

def drive_order(env,order,private=(),before_commit=None):
    s=env.initial();events=[]
    if "AUTHENTICATE_CUSTOMER" in env.options(s):
        s,a=env.transition(s,"AUTHENTICATE_CUSTOMER")
        assert a["verdict"]=="VALID",a
    for f in order:
        if s.native.done:break
        for action in (env.encode_action("PROCESS_"+f),
                       env.encode_action("VERIFY_PRIVATE_"+f) if f in private
                       else env.encode_action("SEND_RAW_"+f)):
            if action not in env.options(s):
                raise AssertionError("Not an offered action: "+action+str(env.options(s)))
            n,a=env.transition(s,action)
            events.append({"action":action,"audit":a})
            if a["verdict"]!="VALID":return n,events,a
            s=n
            if s.native.done:break
        if s.native.done:break
        if f in private:
            s,a=env.transition(s,env.encode_action("SEND_ATTEST_"+f))
            events.append({"action":"SEND_ATTEST_"+f,"audit":a})
            if a["verdict"]!="VALID":return s,events,a
    if "REQUEST_CUSTOMER_CONFIRMATION" in env.options(s):
        s,a=env.transition(s,"REQUEST_CUSTOMER_CONFIRMATION")
        events.append({"action":"REQUEST_CUSTOMER_CONFIRMATION","audit":a})
        if a["verdict"]!="VALID":return s,events,a
    if before_commit:before_commit(s)
    if "COMPLETE_BUSINESS_TASK" in env.options(s):
        s,a=env.transition(s,"COMPLETE_BUSINESS_TASK")
        events.append({"action":"COMPLETE_BUSINESS_TASK","audit":a})
        return s,events,a
    return s,events,{"verdict":"NOT_COMPLETE"}

class GroundedTests(unittest.TestCase):
    def mk(self,root="W01",history="linked",allowed=False,profile="full"):
        return GroundedPublicCIEnv(LOOKUP[root],history,allowed,profile)

    def test_safe_full_task_writes_real_store_in_both_domains(self):
        for root in ("W01","R03"):
            env=self.mk(root)
            proof=find_safe_witness(env.knowledge)
            st,log,last=drive_order(env,proof["order"],proof["privately_verified"])
            self.assertEqual(last["verdict"],"VALID")
            self.assertTrue(st.native.success)
            self.assertFalse(st.native.violation)
            if root=="W01":
                self.assertEqual(st.store["calendar"]["new_lunch"]["participants"],
                                 ["colleague@example.test"])
                self.assertEqual(st.store["calendar"]["new_lunch"]["time"],"13:00")
            else:
                self.assertEqual(st.store["order"]["item"],"shirt_green_M")
                self.assertEqual(st.store["order"]["status"],"pending (items modified)")
                self.assertEqual(st.store["order"]["total"],36.0)
                self.assertEqual(st.store["gift_cards"]["gift_card"],14.0)
                self.assertTrue(st.confirmed)
                self.assertEqual(st.store["modification_count"],1)

    def test_flip_hold_and_permission_same_current_message(self):
        for root in ("W01","R03"):
            audits={}
            for history,allowed in (
                ("linked",False),("unrelated",False),("hold_a",False),
                ("hold_b",False),("linked",True)):
                env=self.mk(root,history,allowed,"minimal")
                st=env.initial()
                if "AUTHENTICATE_CUSTOMER" in env.options(st):
                    st,a=env.transition(st,"AUTHENTICATE_CUSTOMER")
                    self.assertEqual(a["verdict"],"VALID")
                f=env.seed.anchor
                st,a=env.transition(st,env.encode_action("PROCESS_"+f))
                self.assertEqual(a["verdict"],"VALID",a)
                obs=env.observation(st)
                self.assertTrue(any("source_document" in x["message"]
                                    for x in obs["tool_results"]))
                st,a=env.transition(st,env.encode_action("SEND_RAW_"+f))
                self.assertEqual(a.get("provenance_verified"),True)
                audits[(history,allowed)]=a
            self.assertEqual(audits["linked",False]["verdict"],"VIOLATION")
            self.assertEqual(audits["unrelated",False]["verdict"],"VALID")
            self.assertEqual(audits["hold_a",False]["verdict"],"VALID")
            self.assertEqual(audits["hold_b",False]["verdict"],"VALID")
            self.assertEqual(audits["linked",True]["verdict"],"VALID")
            self.assertTrue(audits["linked",True]["new_derived_private"])
            self.assertEqual(audits["linked",False]["supporting_facts"],
                             audits["linked",True]["supporting_facts"])

    def test_no_typed_fact_leak_in_observation(self):
        for root in ("W01","R03"):
            env=self.mk(root)
            st=env.initial()
            if "AUTHENTICATE_CUSTOMER" in env.options(st):
                st,a=env.transition(st,"AUTHENTICATE_CUSTOMER")
            obs=env.observation(st)
            for forbidden in ("native","root_source_facts","source_certificate",
                              "issued","support","f0","f1","f2","f3",
                              "ground_truth","reward","violation","seed_id"):
                self.assertNotIn(forbidden,obs)
            self.assertTrue(obs["request"])
            self.assertNotIn("grounded_fact_internal",json.dumps(obs))

    def test_all_grounded_facts_bidirectional_text_roundtrip(self):
        for root in ("W01","R03"):
            env=self.mk(root)
            st=env.initial()
            claims=root_source_facts(env.spec,st.store)
            for f in FACTS:
                claim=claims[f]
                rendered=safe_text_assertion(claim)
                self.assertEqual(parse_text_assertion(rendered),
                                 normalized_claim(claim))
                self.assertNotIn(f,rendered)

    def test_unverified_or_tampered_tool_message_returns_unknown_without_advance(self):
        for root in ("W01","R03"):
            env=self.mk(root,profile="minimal")
            st=env.initial()
            if "AUTHENTICATE_CUSTOMER" in env.options(st):
                st,a=env.transition(st,"AUTHENTICATE_CUSTOMER")
                self.assertEqual(a["verdict"],"VALID")
            fact=env.seed.anchor
            before_hash=digest(asdict(st.native))
            st,a=env.transition(st,env.encode_action("SEND_RAW_"+fact))
            self.assertEqual(a["verdict"],"INVALID_ACTION") # source read required by phase
            self.assertEqual(before_hash,digest(asdict(st.native)))
            st,a=env.transition(st,env.encode_action("PROCESS_"+fact))
            self.assertEqual(a["verdict"],"VALID")
            original=digest(asdict(st.native))
            proof=deepcopy(st)
            result=proof.tool_results[-1]
            decoded=json.loads(result["message"])
            decoded["authorized_source_document"]="This record explicitly denies any link to a case."
            result["message"]=json.dumps(decoded)
            unchanged,err=env.transition(proof,env.encode_action("SEND_RAW_"+fact))
            self.assertEqual(err["verdict"],"UNKNOWN")
            self.assertTrue(err["task_evidence_rejected"])
            self.assertEqual(digest(asdict(unchanged.native)),original)
            self.assertFalse(unchanged.native.violation)
            self.assertFalse(unchanged.native.success)
            # Change factual source record after reading; stale credit cannot deliver.
            stale=deepcopy(st)
            if root=="W01":
                stale.store["contacts"]["employee_record"]["case_id"]="WRONG-REFERENCE"
            else:
                stale.store["order"]["case_reference"]="WRONG-REFERENCE"
            _,err=env.transition(stale,env.encode_action("SEND_RAW_"+fact))
            self.assertEqual(err["verdict"],"UNKNOWN")

    def test_delivered_messages_are_immutable_after_store_changes(self):
        for root in ("W01","R03"):
            env=self.mk(root,"unrelated",False,"full")
            st=env.initial()
            if "AUTHENTICATE_CUSTOMER" in env.options(st):
                st,a=env.transition(st,"AUTHENTICATE_CUSTOMER")
                self.assertEqual(a["verdict"],"VALID")
            f=env.seed.anchor
            st,a=env.transition(st,env.encode_action("PROCESS_"+f))
            self.assertEqual(a["verdict"],"VALID")
            st,a=env.transition(st,env.encode_action("SEND_RAW_"+f))
            self.assertEqual(a["verdict"],"VALID")
            before=[
                x["message"] for x in env.observation(st)["previously_delivered_messages"]
                if x["recipient"]=="R"]
            self.assertTrue(any(x.startswith("Verified record:") for x in before))
            shifted=deepcopy(st)
            if root=="W01":
                shifted.store["contacts"]["employee_record"]["case_id"]="ANOTHER-CASE"
            else:
                shifted.store["order"]["case_reference"]="ANOTHER-CASE"
            after=[
                x["message"] for x in env.observation(shifted)["previously_delivered_messages"]
                if x["recipient"]=="R"]
            self.assertEqual(before,after)

    def test_w01_missing_contact_and_calendar_conflict(self):
        env=self.mk("W01")
        proof=find_safe_witness(env.knowledge)
        # Source invalidated before the first tool read.
        st=env.initial()
        st.store["contacts"].pop("collaborator")
        action=env.encode_action("PROCESS_f1")
        self.assertIn(action,env.options(st))
        n,a=env.transition(st,action)
        self.assertEqual(a["verdict"],"UNKNOWN")
        self.assertEqual(digest(asdict(st.native)),digest(asdict(n.native)))
        self.assertFalse(n.native.success)
        # The entire safe plan must not commit on a calendar conflict.
        env2=self.mk("W01")
        orig=env2.initial
        def bad_initial():
            st=orig()
            st.store["calendar"]["lunch_conflict"]={
                "title":"Existing appointment","date":"2026-11-25",
                "time":"13:15","minutes":60,"participants":["owner@example.test"],
                "status":"scheduled"}
            return st
        env2.initial=bad_initial
        st,log,last=drive_order(env2,proof["order"],proof["privately_verified"])
        self.assertEqual(last["verdict"],"UNKNOWN")
        self.assertFalse(st.native.success)
        self.assertNotIn("new_lunch",st.store["calendar"])

    def test_r03_unavailable_item_and_payment_guard(self):
        env=self.mk("R03")
        proof=find_safe_witness(env.knowledge)
        orig=env.initial
        for change in ("unavailable","bad_product","no_balance",
                       "bad_status","different_owner"):
            def mutated():
                st=orig()
                if change=="unavailable":
                    st.store["catalog"]["shirt_green_M"]["available"]=False
                elif change=="bad_product":
                    st.store["catalog"]["shirt_green_M"]["product_id"]="shoes"
                elif change=="no_balance":
                    st.store["gift_cards"]["gift_card"]=1.0
                elif change=="bad_status":
                    st.store["order"]["status"]="delivered"
                else:
                    st.store["order"]["owner_id"]="another_user"
                return st
            env.initial=mutated
            if change=="different_owner":
                initial=env.initial()
                nxt,a=env.transition(initial,"AUTHENTICATE_CUSTOMER")
                self.assertEqual(a["verdict"],"UNKNOWN")
                self.assertFalse(nxt.native.success)
                continue
            st,events,last=drive_order(env,proof["order"],proof["privately_verified"])
            self.assertFalse(st.native.success,change)
            self.assertFalse(st.native.violation,change)
            self.assertNotEqual(last["verdict"],"VALID",change)
            self.assertNotEqual(st.store["order"]["status"],"pending (items modified)")
        env.initial=orig

    def test_r03_user_refusal_and_wrong_confirmation_binding(self):
        env=self.mk("R03")
        proof=find_safe_witness(env.knowledge)
        orig=env.initial
        for reply,binding in (("no","current_update"),("yes","wrong_proposal")):
            def changed_initial():
                st=orig()
                st.store["customer_reply"]=reply
                st.store["confirmation_binding"]=binding
                return st
            env.initial=changed_initial
            st,events,last=drive_order(env,proof["order"],proof["privately_verified"])
            self.assertFalse(st.confirmed)
            self.assertFalse(st.native.success)
            self.assertEqual(st.store["order"]["item"],"shirt_blue_M")
            self.assertNotIn("COMPLETE_BUSINESS_TASK", [e["action"] for e in events])
            self.assertEqual(env.options(st),("ABORT",))
        env.initial=orig

    def test_auth_mismatch_and_direct_completion_denied(self):
        env=self.mk("R03")
        st=env.initial()
        st.store["customer"]["zip"]="99999"
        n,a=env.transition(st,"AUTHENTICATE_CUSTOMER")
        self.assertEqual(a["verdict"],"UNKNOWN")
        self.assertFalse(n.store["customer"]["authenticated"])
        good=env.initial()
        good,a=env.transition(good,"AUTHENTICATE_CUSTOMER")
        self.assertEqual(a["verdict"],"VALID")
        n,a=env.transition(good,"COMPLETE_BUSINESS_TASK")
        self.assertEqual(a["verdict"],"INVALID_ACTION")
        self.assertFalse(n.native.success)

    def test_original_public_ci_engine_remains_a_separate_version(self):
        from kci.public_workflow_ci import PublicCIEnv
        self.assertNotEqual(PublicCIEnv,GroundedPublicCIEnv)
        for root in ("W01","R03"):
            old=PublicCIEnv(LOOKUP[root],"linked",False,"minimal")
            new=GroundedPublicCIEnv(LOOKUP[root],"linked",False,"minimal")
            self.assertEqual(old.seed,new.seed)
            self.assertNotEqual(old.version if hasattr(old,"version") else "legacy",new.version)

if __name__=="__main__":unittest.main()
