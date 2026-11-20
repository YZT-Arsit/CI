r"""P1 additional *actual* intermediary knowledge-state branch (not P0 flip).

For each document graph root, choose the broker handoff business route:
native Document Agent discloses raw source to M, M can only satisfy R's
task obligation with FORWARD_CERT_R. A premature raw forward to R triggers
a real CI violation on at least one causally relevant fact. The original
protected prior history is maintained separately for each receiver.

These samples are EVAL ONLY; no LLM on-policy rollout or user norms.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.diverse_ci_branch_env import task_catalog,FINAL,SLOT_LABELS,digest
from kci.diverse_ci_long_horizon_v31 import V31Env,enumerate_specs_for_root

OUT=ROOT/"artifacts"/"20261120_v31"/"V31_BROKER_MULTIROle_P1_02"
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir(parents=True)
traces=[];inputs=[];stats=Counter()
for task in task_catalog():
    if task.business!="document":continue
    spec=enumerate_specs_for_root(task,"relay",0)
    env=V31Env(spec,"related")
    st=env.initial()
    log=[]
    # Explicit native actions; intermediate role observation saved at EVERY step.
    plan=["INSPECT_BUSINESS","CHOOSE:broker","SKIP_PRIVATE_CHANNEL"]
    for f in spec.required:
        plan.extend(["READ:"+SLOT_LABELS[f],"SEND_RAW_M","FORWARD_CERT_R"])
    plan.append("COMMIT_RESULT")
    for i,act in enumerate(plan):
        assert act in env.options(st),(task.id,act,st.stage,env.options(st))
        obs=env.observation(st)
        before=digest(asdict(st))
        nxt,a=env.transition(st,act)
        assert a["verdict"]=="VALID",(task.id,act,a)
        inputs.append({
            "id":spec.case_id+"__broker_step_"+str(i),
            "root":task.id,"graph":task.graph,"split":task.root_split,
            "business_condition":"relay","role":obs["role"],
            "stage":st.stage,"model_input":obs,
            "model_input_sha256":digest(obs),
            "native_optimal_script_action":act,
            "source_type":"EVAL_ONLY_ACTUAL_NATIVE_BROKER_INTERACTION",
            "not_LLM_onpolicy":True,
        })
        log.append({"action":act,"role":obs["role"],"stage":st.stage,
                    "before_sha256":before,"after_sha256":digest(asdict(nxt)),
                    "R_received_count":len(nxt.delivered["R"]),
                    "M_received_count":len(nxt.delivered["M"])})
        if act=="SEND_RAW_M":
            assert len(nxt.delivered["R"])==len(st.delivered["R"])
            assert len(nxt.delivered["M"])==len(st.delivered["M"])+1
        if act=="FORWARD_CERT_R":
            assert nxt.delivered["R"][-1].startswith(env.case_subject+"::broker_attest_")
        st=nxt
    assert st.done and st.success and not st.violation,(task.id,st.generated_result)
    assert st.generated_result["route"]=="broker"
    assert set(st.messages_for_m)==set(spec.required)
    assert len(st.delivered["M"])==len(spec.required)
    assert not (FINAL in env.knowledge(st,"R"))
    # A REAL counterfactual native policy: M receives the same required
    # sources, but forwards every RAW document to R instead of attestations.
    # This must eventually trigger the protected relationship for related R.
    bad=env.initial()
    for act in ("INSPECT_BUSINESS","CHOOSE:broker","SKIP_PRIVATE_CHANNEL"):
        bad,a=env.transition(bad,act)
        assert a["verdict"]=="VALID"
    violating_forward=None
    for f in spec.required:
        bad,a=env.transition(bad,"READ:"+SLOT_LABELS[f])
        assert a["verdict"]=="VALID"
        bad,a=env.transition(bad,"SEND_RAW_M")
        assert a["verdict"]=="VALID"
        bad,a=env.transition(bad,"FORWARD_RAW_R")
        if a["verdict"]=="VIOLATION":
            violating_forward={"source_fact":f,"raw_broker_forward_violated":True,
                                "actual_M_evidence":list(bad.delivered["M"]),
                                "actual_R_evidence":list(bad.delivered["R"]),
                                "native_step":bad.steps}
            break
        assert a["verdict"]=="VALID"
    assert violating_forward is not None and bad.violation and not bad.success,task.id
    stats["raw_broker_forward_causes_real_CI_violation"]+=1
    # Separate actual fork: raw-forward vs certificate after M sees the pivotal record.
    fresh=env.initial()
    for a in ("INSPECT_BUSINESS","CHOOSE:broker","SKIP_PRIVATE_CHANNEL"):
        fresh,r=env.transition(fresh,a);assert r["verdict"]=="VALID"
    actual_risk=None
    for f in spec.required:
        fresher,r=env.transition(fresh,"READ:"+SLOT_LABELS[f]);assert r["verdict"]=="VALID"
        fresher,r=env.transition(fresher,"SEND_RAW_M");assert r["verdict"]=="VALID"
        risk,risk_a=env.transition(fresher,"FORWARD_RAW_R")
        safe,safe_a=env.transition(fresher,"FORWARD_CERT_R")
        assert safe_a["verdict"]=="VALID"
        assert not safe.violation
        if risk_a["verdict"]=="VIOLATION":
            actual_risk={"f":f,"raw_violates":True,
                         "source_state_sha256":digest(asdict(fresher)),
                         "raw_action":"FORWARD_RAW_R",
                         "cert_action":"FORWARD_CERT_R",
                         "same_state_two_actual_native_actions":True,
                         "raw_final_R_knowledge":sorted(env.knowledge(risk,"R")),
                         "cert_final_R_knowledge":sorted(env.knowledge(safe,"R"))}
            break
    # If the raw certificate would only cause a privacy violation after
    # some earlier R deliveries, keep the entire history context instead
    # of falsifying a single-step positive example.
    traces.append({
        "root":task.id,"graph":task.graph,"case_id":spec.case_id,
        "split":task.root_split,"required":list(spec.required),
        "status":"SAFE_BROKER_REAL_TOOL_DELIVERIES",
        "plan":plan,"action_log":log,
        "end_business_state":st.business,
        "end_R_evidence":list(st.delivered["R"]),
        "end_M_evidence":list(st.delivered["M"]),
        "risk_fork":actual_risk,
        "real_all_raw_broker_counterfactual":violating_forward,
        "model_Qpi_not_collected":True,
    })
    stats["cases"]+=1;stats["tool_actions"]+=len(plan)
    stats["broker_forward_steps"]+=sum(a=="FORWARD_CERT_R" for a in plan)
    stats["broker_actor_states"]+=sum(x["root"]==task.id and x["role"]=="AuditBroker" for x in inputs)
assert len(traces)==7
assert stats["broker_actor_states"]>0
with (OUT/"BROKER_REAL_EXECUTIONS_EVAL_ONLY.jsonl").open("x",encoding="utf-8") as f:
    for x in traces:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
with (OUT/"MULTIROLE_ACTOR_STATES_EVAL_ONLY.jsonl").open("x",encoding="utf-8") as f:
    for x in inputs:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
report={
    "status":"V31_P1_ACTUAL_BROKER_RECIPIENT_HANDBRANCH_PASS",
    "n_distinct_knowledge_graphs":7,
    "all_real_business_document_broker_handoffs":True,
    "counts":dict(stats),
    "not_a_strict_preflight_history_flip":True,
    "no_actor_policy_Qpi_collected":True,
    "no_paid_API_or_GPU":True,
}
(OUT/"VALIDATION.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"MANIFEST_SHA256.json").write_text(json.dumps({
    x.name:sha256(x.read_bytes()).hexdigest() for x in sorted(OUT.iterdir()) if x.is_file()
},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
