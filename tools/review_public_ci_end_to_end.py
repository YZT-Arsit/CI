"""Safe-path end-to-end intervention audit, all ten synthetic business workflows.

The intervention changes a clone of each task's initial business fixture before
any tool call, then replays the real workflow via the original simulator.
It never edits stored datasets or official external benchmarks.
"""
from pathlib import Path
from copy import deepcopy
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci import PUBLIC_TASKS,PublicCIEnv,check_goal
from kci.knowledge_world_seed_v2 import find_safe_witness
DATA=ROOT/"artifacts"/"20261119_composition_signals"/"public_workflow_ci_10_final"
out=[]
for sp in PUBLIC_TASKS:
    env=PublicCIEnv(sp,"linked",False,"full")
    safe=find_safe_witness(env.knowledge)
    assert safe is not None
    baseline=env.run_script(safe["order"],safe["privately_verified"])
    assert baseline["success"]
    orig=env.initial
    def inject():
        s=orig()
        st=s.store
        key=sp.action_type
        if key=="calendar_create":
            st["contacts"]={}
        elif key=="calendar_reschedule":
            st["calendar"]["event_A"]["title"]="Different unrequested meeting"
        elif key=="calendar_invite":
            st["contacts"]={}
        elif key=="send_task_emails":
            st["files"]["todo"]="No named owners, no assigned deadlines, no work items."
        elif key=="create_and_share":
            st["files"]["source_note"]="Only pack a first aid kit. Do not pack electronics."
        elif key=="retail_exchange":
            st["catalog"]["shirt_green_M"]["available"]=False
        elif key=="retail_return":
            st["order"]["payment"]="unsupported_unknown_wallet"
        elif key=="retail_modify_item":
            st["catalog"]["shirt_green_M"]["available"]=False
        elif key=="retail_address":
            st["customer"]["name"]="Entirely different customer"
            st["customer"]["zip"]="99999"
        elif key=="retail_cancel":
            st["order"]["cancel_reason"]="reason not provided by customer"
        else:raise RuntimeError(key)
        return s
    env.initial=inject
    try:
        r=env.run_script(safe["order"],safe["privately_verified"])
        passed=r["success"] and not r["violation"]
        error=None
    except Exception as e:
        passed=False
        error=type(e).__name__+": "+str(e)
    out.append({
        "id":sp.id,
        "business_goal":sp.user_request,
        "baseline_safe_success":baseline["success"],
        "intervention_case":sp.action_type,
        "edited_pretool_store_only":True,
        "safe_success_after_incompatible_tool_source_fixture":passed,
        "error":error})
results={
    "test":"Full tool-execution safe path replay after one synthetic business-data intervention",
    "n":len(out),
    "remains_safe_success_after_intervention":sum(x["safe_success_after_incompatible_tool_source_fixture"] for x in out),
    "rows":out,
    "caveat":"Different interventions are specific negative controls, not a matched severity scale; some synthetic business contracts do not formally prohibit all interventions. These are fidelity red flags, not proof that the official benchmark backend has a bug."
}
p=DATA/"FIRST_PASS_END_TO_END_BUSINESS_INTERVENTION.json"
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
