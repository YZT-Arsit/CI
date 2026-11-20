"""V3.1 Gate A regression on structured same-entity normal facts and true Q* flip."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import sys,re,unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,SLOT_LABELS,BASE_FACTS,FINAL,closure,digest
from kci.diverse_ci_long_horizon_v3 import LongHorizonFlipCIEnv
from kci.diverse_ci_long_horizon_v31 import (
    V31Env,enumerate_specs_for_root,permuted_rules)
from solve_v31_longterm import pair_values,before_preflight,reward
from audit_kci_v31_gateA import choose_lexical3,choose_observable_rules

def specimen(n=0):
    t=task_catalog()[n]
    condition={"calendar":"early_open","retail":"pending_yes","document":"external"}[t.business]
    return enumerate_specs_for_root(t,condition,n%3)

class V31GateTests(unittest.TestCase):
    def test_seven_rule_families_do_not_touch_frozen_v3(self):
        base={x.graph for x in task_catalog()}
        self.assertEqual(len(base),7)
        for name in base:
            t=next(x for x in task_catalog() if x.graph==name)
            e=V31Env(enumerate_specs_for_root(t,{
                "calendar":"early_open","retail":"pending_yes","document":"external"
            }[t.business],0))
            self.assertEqual(e.version,"kci_diverse_long_horizon_v3_1_mechanism_pilot")
            self.assertNotEqual(type(e),LongHorizonFlipCIEnv)

    def test_only_normal_fact_ids_used_for_same_entity_positive_and_negative(self):
        for i in (0,4,8,12,16,20):
            spec=specimen(i)
            yes=V31Env(spec,"related")
            no=V31Env(spec,"same_entity_safe")
            a=yes.initial();b=no.initial()
            self.assertEqual(len(a.ledger),len(b.ledger))
            self.assertTrue(len(a.ledger)>0)
            self.assertTrue(all(y["sender"]=="Archive" for y in a.ledger+b.ledger))
            self.assertTrue(all("S07" not in y["text"] and "OTHER-" not in y["text"]
                                for y in a.ledger+b.ledger))
            self.assertTrue(all(y["fact"].startswith(yes.case_subject+"::")
                                for y in a.ledger+b.ledger))
            self.assertEqual(
                set(x["fact"].split("::",1)[1] for x in a.ledger),set(spec.risky_prior))
            self.assertEqual(
                set(x["fact"].split("::",1)[1] for x in b.ledger),set(spec.safe_prior))
            self.assertTrue(set(spec.risky_prior+spec.safe_prior)<=set(BASE_FACTS))

    def test_actor_cannot_infer_label_from_existing_hidden_fields(self):
        e=V31Env(specimen(0),"same_entity_safe")
        state,visible=before_preflight(e)
        self.assertEqual(visible["role"],"Coordinator")
        self.assertIn("declared_knowledge_rules",visible)
        self.assertIn("task_evidence_contract",visible)
        self.assertIn("current_case_subject",visible)
        for k in ("target_action","Qstar","oracle","violation","reward",
                  "history_kind","positive_history","native_state"):
            self.assertNotIn(k,visible)
        self.assertTrue(visible["delivered_message_log"])
        self.assertEqual(len(visible["task_evidence_contract"]["required_slots"]),
                         len(e.required_facts))

    def test_history_is_only_difference_in_real_actor_prompt(self):
        for i in (0,5,10,14,19):
            s=specimen(i)
            a=V31Env(s,"related")
            b=V31Env(s,"same_entity_safe")
            o1=before_preflight(a)[1].copy()
            o2=before_preflight(b)[1].copy()
            left=o1.pop("delivered_message_log")
            right=o2.pop("delivered_message_log")
            self.assertEqual(o1,o2)
            self.assertNotEqual(left,right)
            self.assertEqual(len(left),len(right))
            self.assertEqual(len(before_preflight(a)[1]["legal_actions"]),3)

    def test_relevant_history_causes_strict_long_horizon_preflight_value_flip(self):
        for i in (0,4,8,12,16,20):
            with self.subTest(graph=task_catalog()[i].graph):
                s=specimen(i)
                yes=pair_values(V31Env(s,"related"))
                no=pair_values(V31Env(s,"same_entity_safe"))
                self.assertGreater(yes["Qgap_open_minus_skip"],0)
                self.assertLess(no["Qgap_open_minus_skip"],0)
                self.assertTrue(yes["open"]["terminal"]["success"])
                self.assertFalse(yes["skip"]["terminal"]["success"])
                self.assertTrue(no["skip"]["terminal"]["success"])
                self.assertEqual(yes["model_observation"]["legal_actions"],
                                 no["model_observation"]["legal_actions"])

    def test_other_entity_histories_never_falsely_join_private_rule(self):
        for i in (0,6,12,18):
            spec=specimen(i)
            env=V31Env(spec,"cross_entity_related")
            e=env.initial()
            assert e.ledger
            self.assertTrue(all(
                x["fact"].startswith(env.other_subject+"::") for x in e.ledger))
            src=pair_values(env)
            self.assertLess(src["Qgap_open_minus_skip"],0)
            self.assertFalse(src["skip"]["terminal"]["violation"])

    def test_full_visible_rule_planner_beats_independent_lexical_three_condition_rule(self):
        wrong=0;total=0
        for i in (0,4,8,12,16,20):
            s=specimen(i)
            for h in ("related","same_entity_safe","cross_entity_related"):
                e=V31Env(s,h)
                true=pair_values(e)
                gold=("OPEN_PRIVATE_CHANNEL" if true["Qgap_open_minus_skip"]>0
                      else "SKIP_PRIVATE_CHANNEL")
                obs=true["model_observation"]
                self.assertEqual(choose_observable_rules(obs),gold)
                wrong+=int(choose_lexical3(obs)!=gold)
                total+=1
        self.assertGreater(wrong,0)
        self.assertLess(wrong,total)

    def test_different_task_lengths_and_real_costs_produce_distinct_gap_magnitudes(self):
        positives=set();negatives=set();nreq=set();fees=set();costs=set()
        for i,t in enumerate(task_catalog()[:12]):
            for j in (0,1,2):
                cond={"calendar":"early_open","retail":"pending_yes","document":"external"}[t.business]
                s=enumerate_specs_for_root(t,cond,j)
                nreq.add(len(s.required));fees.add(s.prep_fee);costs.add(s.cost_per_step)
                yes=pair_values(V31Env(s,"related"))
                no=pair_values(V31Env(s,"same_entity_safe"))
                positives.add(yes["Qgap_open_minus_skip"])
                negatives.add(no["Qgap_open_minus_skip"])
        self.assertEqual(nreq,{1,2,3})
        self.assertEqual(fees,{2,3,4})
        self.assertGreaterEqual(len(costs),4)
        self.assertGreaterEqual(len(positives),5)
        self.assertGreaterEqual(len(negatives),4)

    def test_native_two_action_trace_is_real_legal_path(self):
        env=V31Env(specimen(13),"related")
        p=pair_values(env)
        st,_=before_preflight(env)
        for arm in (p["open"],p["skip"]):
            current=st
            for x in arm["action_trace"]:
                self.assertEqual(x["state_before_sha256"],digest(asdict(current)))
                nxt,a=env.transition(current,x["action"])
                self.assertEqual(a["verdict"],x["actual_verdict"])
                current=nxt
            self.assertTrue(current.done)
            self.assertEqual(reward(current,env.spec.cost_per_step),
                             arm["native_exact_Qstar"])

    def test_tampered_certified_source_document_fails_closed(self):
        env=V31Env(specimen(3),"related")
        st,_=before_preflight(env)
        st,_=env.transition(st,"OPEN_PRIVATE_CHANNEL")
        f=env.required_facts[0]
        st,_=env.transition(st,"READ:"+SLOT_LABELS[f])
        forged=deepcopy(st)
        forged.tool_results[-1]["source_document"]="No identifying fact was disclosed."
        expected=digest(asdict(forged))
        nxt,report=env.transition(forged,"SEND_RAW_R")
        self.assertEqual(report["verdict"],"UNKNOWN")
        self.assertEqual(digest(asdict(nxt)),expected)

    def test_no_conflation_of_history_and_norm_authorization(self):
        s=specimen(0)
        x=pair_values(V31Env(s,"related"))
        y=pair_values(V31Env(s,"authorized_related"))
        self.assertGreater(x["Qgap_open_minus_skip"],0)
        self.assertLess(y["Qgap_open_minus_skip"],0)
        self.assertNotEqual(x["model_observation"]["recipient_policy"],
                            y["model_observation"]["recipient_policy"])
        self.assertEqual(x["model_observation"]["delivered_message_log"],
                         y["model_observation"]["delivered_message_log"])

if __name__=="__main__":unittest.main()
