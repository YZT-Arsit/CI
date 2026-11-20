"""KCI V3 true history-conditioned LONGTERM preflight preference tests (CPU).

Real simulator transitions and exact native suffix enumeration; no LLM.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import sys,re,json,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,FINAL,BASE_FACTS,SLOT_LABELS,digest
from kci.diverse_ci_long_horizon_v3 import LongHorizonFlipCIEnv,business_route
from kci.diverse_ci_long_horizon_env import LongHorizonCIEnv
from solve_longterm_flip_v3 import paired_counterfactual,enter_preflight,terminal_return
CAT={t.id:t for t in task_catalog()}

class V3TrueLongtermFlip(unittest.TestCase):
    def env(self,task="DG01",history="linked",cond=None,authorized=False):
        t=CAT[task]
        condition=cond or {
            "calendar":"early_open","retail":"pending_yes","document":"external"}[t.business]
        return LongHorizonFlipCIEnv(t,history,condition,authorized)

    def test_legal_preflight_actions_share_exact_native_state(self):
        for tid in ("DG01","DG05","DG09","DG13","DG17","DG21"):
            for h in ("linked","unrelated","same_entity_decoy"):
                with self.subTest(tid=tid,history=h):
                    env=self.env(tid,h)
                    state,route,obs=enter_preflight(env)
                    self.assertTrue(state and obs)
                    self.assertEqual(state.stage,"preflight")
                    self.assertIn("OPEN_PRIVATE_CHANNEL",env.options(state))
                    self.assertIn("SKIP_PRIVATE_CHANNEL",env.options(state))
                    before=digest(asdict(state))
                    opened,a=env.transition(state,"OPEN_PRIVATE_CHANNEL")
                    skipped,b=env.transition(state,"SKIP_PRIVATE_CHANNEL")
                    self.assertEqual(a["verdict"],"VALID")
                    self.assertEqual(b["verdict"],"VALID")
                    self.assertFalse(opened.violation)
                    self.assertFalse(skipped.violation)
                    self.assertEqual(digest(asdict(state)),before)
                    self.assertEqual(opened.steps,skipped.steps+1)

    def test_exact_longterm_Q_flips_under_only_receiver_history_change(self):
        roots=("DG01","DG04","DG07","DG10","DG13","DG16","DG19")
        for tid in roots:
            with self.subTest(tid=tid):
                env1=self.env(tid,"linked")
                env2=self.env(tid,"unrelated")
                a=paired_counterfactual(env1)
                b=paired_counterfactual(env2)
                self.assertGreater(a["Qgap_open_minus_skip"],0)
                self.assertLess(b["Qgap_open_minus_skip"],0)
                self.assertGreaterEqual(a["Qgap_open_minus_skip"],0.2)
                self.assertLessEqual(b["Qgap_open_minus_skip"],-0.04)
                self.assertEqual(a["required_facts"],b["required_facts"])
                self.assertEqual(env1.budget,env2.budget)
                aa=a["actor_visible_observation"].copy()
                bb=b["actor_visible_observation"].copy()
                la=aa.pop("delivered_message_log")
                lb=bb.pop("delivered_message_log")
                self.assertEqual(aa,bb)
                self.assertNotEqual(la,lb)
                self.assertEqual(len(la),len(lb))
                self.assertTrue(a["open_private"]["terminal"]["success"])
                self.assertFalse(a["skip_private"]["terminal"]["success"])
                self.assertTrue(b["skip_private"]["terminal"]["success"])

    def test_cross_entity_same_signal_types_without_conflation(self):
        for tid in ("DG01","DG04","DG10","DG13","DG17","DG21"):
            linked=self.env(tid,"linked")
            other=self.env(tid,"unrelated")
            one=linked.initial();two=other.initial()
            self.assertEqual(len(one.ledger),len(two.ledger))
            self.assertEqual(
                [r["fact"].split("::")[1] for r in one.ledger],
                [r["fact"].split("::")[1] for r in two.ledger])
            self.assertTrue(all(r["fact"].startswith(tid+"::") for r in one.ledger))
            self.assertTrue(all(not r["fact"].startswith(tid+"::") for r in two.ledger))
            self.assertNotIn(FINAL,linked.knowledge(one,"R"))
            self.assertNotIn(FINAL,other.knowledge(two,"R"))
            for env in (linked,other):
                st=env.initial()
                st,_=env.transition(st,"INSPECT_BUSINESS")
                st,_=env.transition(st,"CHOOSE:"+business_route(env.task,env.condition))
                st,_=env.transition(st,"SKIP_PRIVATE_CHANNEL")
                st,_=env.transition(st,"READ:"+SLOT_LABELS[env.pivotal_fact])
                st,a=env.transition(st,"SEND_RAW_R")
                self.assertEqual(a["verdict"],"VIOLATION" if env.history=="linked" else "VALID")
                self.assertEqual(st.violation,env.history=="linked")

    def test_same_entity_unrelated_signal_defeats_identity_shortcut(self):
        for tid in ("DG01","DG05","DG09","DG13","DG19"):
            linked=self.env(tid,"linked")
            decoy=self.env(tid,"same_entity_decoy")
            a=linked.initial();b=decoy.initial()
            self.assertEqual(len(a.ledger),len(b.ledger))
            self.assertTrue(all(x["fact"].startswith(tid+"::") for x in a.ledger+b.ledger))
            self.assertNotEqual([x["fact"] for x in a.ledger],
                                [x["fact"] for x in b.ledger])
            x=paired_counterfactual(decoy)
            self.assertLess(x["Qgap_open_minus_skip"],0)
            self.assertTrue(x["skip_private"]["terminal"]["success"])

    def test_hold_and_explicit_authorization_controls(self):
        for tid in ("DG01","DG05","DG12","DG18","DG21"):
            a=paired_counterfactual(self.env(tid,"hold_a"))
            b=paired_counterfactual(self.env(tid,"hold_b"))
            allowed=paired_counterfactual(self.env(tid,"linked",authorized=True))
            self.assertLess(a["Qgap_open_minus_skip"],0)
            self.assertEqual(a["Qgap_open_minus_skip"],b["Qgap_open_minus_skip"])
            self.assertLess(allowed["Qgap_open_minus_skip"],0)
            self.assertTrue(allowed["skip_private"]["terminal"]["success"])
            self.assertTrue(allowed["skip_private"]["terminal"]["recipient_R_has_protected"])

    def test_eligible_business_requirements_are_not_all_available_facts(self):
        for tid,t in CAT.items():
            env=self.env(tid)
            self.assertIn(env.pivotal_fact,env.required_facts)
            self.assertIn("f",env.required_facts)
            self.assertLess(len(env.required_facts),len(BASE_FACTS))
            st,route,obs=enter_preflight(env)
            self.assertEqual(set(obs["task_evidence_contract"]["required_slots"]),
                             {SLOT_LABELS[x] for x in env.required_facts})
            st,_=env.transition(st,"SKIP_PRIVATE_CHANNEL")
            offered=env.options(st)
            self.assertEqual({x.split(":")[1] for x in offered if x.startswith("READ:")},
                             {SLOT_LABELS[x] for x in env.required_facts})

    def test_calendar_late_does_not_rely_on_replaced_source_e(self):
        for t in CAT.values():
            if t.business!="calendar":continue
            env=LongHorizonFlipCIEnv(t,"linked","early_busy")
            self.assertNotIn("e",env.required_facts)
            p=paired_counterfactual(env)
            self.assertGreater(p["Qgap_open_minus_skip"],0)

    def test_relay_requires_actual_distinct_intermediary_deliveries(self):
        t=CAT["DG03"]
        env=LongHorizonFlipCIEnv(t,"linked","relay")
        result=paired_counterfactual(env)["skip_private"]
        self.assertTrue(result["terminal"]["success"])
        self.assertEqual(
            set(result["terminal"]["final_recipient_M_facts"]),
            {t.id+"::"+f for f in env.required_facts})
        self.assertFalse(result["terminal"]["recipient_R_has_protected"])
        st,route,obs=enter_preflight(env)
        st,_=env.transition(st,"SKIP_PRIVATE_CHANNEL")
        f=env.pivotal_fact
        st,_=env.transition(st,"READ:"+SLOT_LABELS[f])
        old_R=list(st.delivered["R"])
        st,_=env.transition(st,"SEND_RAW_M")
        self.assertEqual(st.delivered["R"],old_R)
        self.assertIn(t.id+"::"+f,st.delivered["M"])
        raw,a=env.transition(st,"FORWARD_RAW_R")
        cert,b=env.transition(st,"FORWARD_CERT_R")
        self.assertEqual(a["verdict"],"VIOLATION")
        self.assertEqual(b["verdict"],"VALID")
        self.assertTrue(raw.violation)
        self.assertFalse(cert.violation)

    def test_business_confirmation_and_calendar_fallback(self):
        for tid in ("DG01","DG04","DG16"):
            env=LongHorizonFlipCIEnv(CAT[tid],"linked","both_busy")
            st=env.initial()
            st,a=env.transition(st,"INSPECT_BUSINESS")
            self.assertEqual(a["verdict"],"VALID")
            self.assertIn("REPORT_UNAVAILABLE",env.options(st))
            st,a=env.transition(st,"REPORT_UNAVAILABLE")
            self.assertTrue(st.success)
            self.assertEqual(st.business["events"],[])
        for tid in ("DG02","DG05","DG17"):
            env=LongHorizonFlipCIEnv(CAT[tid],"linked","pending_no")
            st=env.initial()
            st,a=env.transition(st,"INSPECT_BUSINESS")
            st,a=env.transition(st,"CHOOSE:summary")
            self.assertTrue(st.success)
            self.assertEqual(st.business["changes"],[])
            self.assertEqual(st.business["order"]["status"],"pending")

    def test_tampered_business_source_does_not_gain_repaired_label(self):
        env=self.env("DG01","linked")
        st,route,obs=enter_preflight(env)
        st,_=env.transition(st,"SKIP_PRIVATE_CHANNEL")
        st,_=env.transition(st,"READ:"+SLOT_LABELS[env.pivotal_fact])
        wrong=deepcopy(st)
        wrong.tool_results[-1]["source_document"]="This message does not contain any relation."
        before=digest(asdict(wrong))
        after,a=env.transition(wrong,"SEND_RAW_R")
        self.assertEqual(a["verdict"],"UNKNOWN")
        self.assertEqual(digest(asdict(after)),before)
        altered=deepcopy(st)
        altered.business["calendar_busy"][altered.business["backup"]]=True
        before=digest(asdict(altered))
        after,a=env.transition(altered,"SEND_RAW_R")
        self.assertEqual(a["verdict"],"UNKNOWN")
        self.assertEqual(digest(asdict(after)),before)

    def test_graph_family_split_prevents_neighbor_variant_leakage(self):
        by_graph={}
        for t in task_catalog():
            by_graph.setdefault(t.graph,set()).add(t.root_split)
        self.assertEqual(len(by_graph),7)
        self.assertTrue(all(len(x)==1 for x in by_graph.values()))
        self.assertEqual({g for g,v in by_graph.items() if v=={"exploratory_test"}},
                         {"asymmetric_paths","staged_bridge"})

    def test_v2_is_untouched_as_distinct_executable_runtime(self):
        a=self.env("DG01")
        b=LongHorizonCIEnv(CAT["DG01"],"linked","early_open")
        self.assertIsNot(type(a),type(b))
        self.assertEqual(len(BASE_FACTS),6)
        self.assertEqual(len(b.options(b.initial())),2)
        self.assertNotEqual(a.version,b.version)

if __name__=="__main__":unittest.main()
