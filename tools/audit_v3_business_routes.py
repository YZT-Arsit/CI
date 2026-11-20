"""P1 real task branching, authorization and cross-recipient evidence controls.

Seven knowledge topologies x four genuine alternate business conditions:
- no calendar slot: report unavailable (not fake booking);
- retail refusal: read-only order summary (no order mutation);
- internal document with permitted recipient: skip costly privacy channel;
- external broker document: raw evidence reaches M, not R, with certified
  forward route and REAL broker handoff contract checked at commit.
All evaluations execute actual native actions, not static string labels.
"""
from pathlib import Path
import sys,json
from hashlib import sha256
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,FINAL,digest,SLOT_LABELS
from kci.diverse_ci_long_horizon_v3 import LongHorizonFlipCIEnv
from solve_longterm_flip_v3 import paired_counterfactual

D=ROOT/"artifacts"/"20261119_composition_signals"/"KCI_V3_BRANCH_CONTROLS_01"
if D.exists():raise FileExistsError(D)
D.mkdir(parents=True)
items=[];counts=Counter()
for t in task_catalog():
    cond={"calendar":"both_busy","retail":"pending_no","document":"internal"}[t.business]
    if t.business in ("calendar","retail"):
        env=LongHorizonFlipCIEnv(t,"linked",cond)
        st=env.initial()
        st,a=env.transition(st,"INSPECT_BUSINESS")
        assert a["verdict"]=="VALID"
        action=("REPORT_UNAVAILABLE" if t.business=="calendar"
                else "CHOOSE:summary")
        assert action in env.options(st)
        final,a=env.transition(st,action)
        assert a["verdict"]=="VALID" and final.success and final.done
        assert not final.violation
        if t.business=="calendar":
            assert final.business["events"]==[] and final.business["report_only"]
            assert not final.generated_result["available_slot"]
        else:
            assert final.business["order"]["status"]=="pending"
            assert final.business["changes"]==[]
            assert final.generated_result["non_mutating_resolution"]
        items.append({"root":t.id,"topology":t.graph,"condition":cond,
                      "task_path":action,"success":final.success,
                      "native_steps":final.steps,
                      "actual_business_result":final.generated_result,
                      "final_business_store_sha256":digest(final.business),
                      "no_preflight_needed":True})
        counts["no_mutation_alternative_branch"]+=1
    else:
        for cond in ("internal","relay"):
            env=LongHorizonFlipCIEnv(t,"linked",cond)
            p=paired_counterfactual(env)
            assert p is not None
            assert p["skip_private"]["terminal"]["success"]
            assert p["Qgap_open_minus_skip"]<0
            last=p["skip_private"]["terminal"]
            if cond=="internal":
                assert env._recipient_authorized(env.initial())
                assert last["actual_tool_call_count"]>0
                assert last["task_business_result"]["route"]=="direct"
                counts["authorized_internal_low_budget"]+=1
            else:
                assert last["task_business_result"]["route"]=="broker"
                assert len(last["final_recipient_M_facts"])==len(env.required_facts)
                assert all(t.id+"::"+f in last["final_recipient_M_facts"]
                           for f in env.required_facts)
                assert not last["recipient_R_has_protected"]
                counts["real_separate_intermediary_handoff"]+=1
                # One native source truly sent to M cannot change R until
                # broker forwards it. Use actual legal source/tool actions.
                st=env.initial()
                st,a=env.transition(st,"INSPECT_BUSINESS")
                st,a=env.transition(st,"CHOOSE:broker")
                st,a=env.transition(st,"SKIP_PRIVATE_CHANNEL")
                f=env.pivotal_fact
                st,a=env.transition(st,"READ:"+SLOT_LABELS[f])
                assert a["verdict"]=="VALID"
                before_R=tuple(st.delivered["R"])
                s,a=env.transition(st,"SEND_RAW_M")
                assert a["verdict"]=="VALID"
                assert tuple(s.delivered["R"])==before_R
                assert t.id+"::"+f in s.delivered["M"]
                raw,ra=env.transition(s,"FORWARD_RAW_R")
                cert,ca=env.transition(s,"FORWARD_CERT_R")
                assert ra["verdict"]=="VIOLATION" and raw.violation
                assert ca["verdict"]=="VALID" and not cert.violation
                assert tuple(cert.delivered["R"])!=before_R
                assert t.id+"::"+f not in cert.delivered["R"]
            items.append({
                "root":t.id,"topology":t.graph,"condition":cond,
                "real_branch_route":p["business_route"],
                "legal_action":p["skip_private"]["first_action"],
                "value_gap_open_minus_skip":p["Qgap_open_minus_skip"],
                "actual_native_trace":p["skip_private"]["actual_action_trace"],
                "actual_business_result":last["task_business_result"],
                "final_M_facts":last["final_recipient_M_facts"],
                "final_R_facts":last["final_recipient_R_facts"],
                "success":last["success"],
                "native_steps":last["steps"],
                "true_broker_handoff":cond=="relay",
                "native_optimal_not_Qpi":True})
            counts["authorized_or_broker_branch"]+=1
assert len(items)==28
assert counts["no_mutation_alternative_branch"]==14
assert counts["real_separate_intermediary_handoff"]==7
with (D/"P1_REAL_BUSINESS_BRANCHES_EVAL_ONLY.jsonl").open("x",encoding="utf-8") as f:
    for x in items:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
summary={
    "status":"V3_TRUE_BUSINESS_BRANCHES_AND_BROKER_CHECK_PASS",
    "root_graphs":7,"branch_scenario_instances":28,
    "counts":dict(counts),
    "all_real_task_branches_executed":True,
    "all_change_real_business_or_recipient_state":True,
    "not_performance_of_trained_agent":True,
    "source_norms_synthetic":True,
}
(D/"VALIDATION.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(D/"MANIFEST_SHA256.json").write_text(json.dumps({
    path.name:sha256(path.read_bytes()).hexdigest()
    for path in sorted(D.iterdir()) if path.is_file()
},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
