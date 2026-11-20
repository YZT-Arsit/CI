r"""CPU-only KCI public-workflow policy baseline suite.

The text-rule controller reads only PublicCIEnv.observation, never native
knowledge closure, env.clues, true action labels, or EVAL truth.
All policies act via the real simulated tool/recipient transition.
"""
import argparse,hashlib,json,re,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci import PUBLIC_TASKS,PublicCIEnv
from kci.knowledge_world_seed_v2 import topological_orders,FACTS,digest,canonical,find_safe_witness

def extract_edges(text):
    import re
    edges=[]
    text=text.strip()
    # Exact, auditable lexical extraction from the *visible natural-language
    # source/receipt*, not from the private evaluator's fact IDs.
    r=re.search(r'(?:links|assigns|associates) Case owner Rowan (?:to|with) '
                r'(?:tracking case|case reference|primary case|backup case|service reference)\s+([A-Za-z0-9-]+)',text,re.I)
    if r:edges.append(("Rowan",r.group(1)))
    r=re.search(r'(?:tracking case|primary case|backup case)\s+([A-Za-z0-9-]+)\s+'
                r'(?:to the restricted category|has restricted category|also corresponds to restricted category)',text,re.I)
    if r:edges.append((r.group(1),"RESTRICTED"))
    r=re.search(r'(?:case reference)\s+([A-Za-z0-9-]+)\s+is linked to private reference\s+([A-Za-z0-9-]+)',text,re.I)
    if r:edges.append((r.group(1),r.group(2)))
    r=re.search(r'private reference\s+([A-Za-z0-9-]+)\s+denotes (?:the )?restricted category',text,re.I)
    if r:edges.append((r.group(1),"RESTRICTED"))
    r=re.search(r'service reference\s+([A-Za-z0-9-]+)\s+is associated with private note\s+([A-Za-z0-9-]+)',text,re.I)
    if r:edges.append((r.group(1),r.group(2)))
    r=re.search(r'private note\s+([A-Za-z0-9-]+)\s+reveals (?:the )?restricted category',text,re.I)
    if r:edges.append((r.group(1),"RESTRICTED"))
    return edges

def text_derived_private(messages):
    edges=[]
    for m in messages:edges.extend(extract_edges(m))
    node={"Rowan"}
    for _ in range(8):
        newer=node|{v for u,v in edges if u in node}
        if newer==node:break
        node=newer
    return "RESTRICTED" in node

def text_observer_should_hide(obs):
    policy=obs["fictional_privacy_policy"]
    if "explicitly authorized for the derived relation" in policy:return False
    existing=[r["message"] for r in obs["previously_delivered_messages"]
              if r["recipient"]=="R" and r["delivered"]]
    docs=[json.loads(v["message"]).get("authorized_source_document","")
          for v in obs["tool_results"] if "authorized_source_document" in v["message"]]
    if len(docs)!=1:
        # Multiple tool outputs are visible to one actor; use the most recent
        # document that matches the current check, not earlier documents.
        if not docs:raise RuntimeError("Actor cannot inspect source document before sharing it")
    candidate=docs[-1]
    return text_derived_private(existing+[candidate])

def select(obs,policy):
    options=obs["available_actions"]
    if "AUTHENTICATE_CUSTOMER" in options:return "AUTHENTICATE_CUSTOMER"
    if "REQUEST_CUSTOMER_CONFIRMATION" in options:return "REQUEST_CUSTOMER_CONFIRMATION"
    if "COMPLETE_BUSINESS_TASK" in options:return "COMPLETE_BUSINESS_TASK"
    send=[x for x in options if x.startswith("DISCLOSE_SOURCE_DOCUMENT:")]
    priv=[x for x in options if x.startswith("REQUEST_PRIVATE_VERIFICATION:")]
    if send and priv:
        assert len(send)==len(priv)==1
        if policy=="always_raw":take_private=False
        elif policy=="always_private":take_private=True
        elif policy=="fixed_redact_intake_verification":
            take_private=send[0].split(":")[-1] in ("intake","verification")
        elif policy=="text_only_symbolic":
            take_private=text_observer_should_hide(obs)
        else:raise ValueError("unknown policy")
        return priv[0] if take_private else send[0]
    att=[x for x in options if x.startswith("DELIVER_VERIFIED_ATTESTATION:")]
    if att:return att[0]
    fetch=[x for x in options if x.startswith("RUN_TOOL:")]
    if fetch:
        # Sorting by contract-visible stage label, not by hidden privacy value.
        label_order={"intake":0,"record":1,"verification":2,"approval":3}
        return min(fetch,key=lambda x:(label_order.get(x.split(":")[-1],8),x))
    if "ABORT" in options:return "ABORT"
    raise RuntimeError("No supported option: "+str(options))

def rollout(env,policy):
    st=env.initial()
    trace=[]
    for t in range(30):
        if st.native.done:break
        obs=env.observation(st)
        action=select(obs,policy)
        if action not in env.options(st):raise RuntimeError("Invalid action")
        st,audit=env.transition(st,action)
        trace.append({"step":t,"role":obs["actor"],"action":action,
                      "verdict":audit["verdict"],
                      "observed_prompt_sha256":digest(obs),
                      "knowledge_derived":audit.get("new_derived_private",False)})
    else:raise RuntimeError("Exceeded explicit evaluation horizon")
    return {
        "policy":policy,"root_id":env.spec.id,"history":env.history,
        "profile":env.profile,"success":bool(st.native.success),
        "violation":bool(st.native.violation),
        "business_goal_met":bool(st.native.success),
        "native_steps":st.native.turn,
        "num_model_decision_steps":len(trace),
        "actual_tool_call_count":len(st.tool_calls),
        "trace":trace,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    a=p.parse_args()
    root=a.dataset
    outfile=root/"CPU_BASELINES.json"
    if outfile.exists():raise FileExistsError(outfile)
    rows=[];agg=defaultdict(lambda:dict(success=0,violation=0,safe_incomplete=0,n=0))
    for spec in PUBLIC_TASKS:
        for profile in ("minimal","full"):
            for history in ("linked","unrelated"):
                env=PublicCIEnv(spec,history,False,profile)
                for policy in ("always_raw","always_private",
                               "fixed_redact_intake_verification","text_only_symbolic"):
                    r=rollout(env,policy)
                    rows.append(r)
                    key=(profile,policy)
                    agg[key]["n"]+=1
                    agg[key]["success"]+=int(r["success"] and not r["violation"])
                    agg[key]["violation"]+=int(r["violation"])
                    agg[key]["safe_incomplete"]+=int(not r["success"] and not r["violation"])
    d={
        "study":"KCI public workflows CPU policy baselines",
        "samples":len(rows),
        "method":"simulated tools + native receiver knowledge, no model or oracle access for four scripted policies",
        "reusable_public_tool_names":True,
        "text_controller_uses_only_actor_observation":True,
        "full_results":[{"profile":kind,"policy":policy,**res}
                        for (kind,policy),res in sorted(agg.items())],
        "private_evaluator_not_visible_to_policy":True,
        "caution":"text-only parser is hand coded against the pilot's controlled wording; a perfect score demonstrates shortcut susceptibility, not general NLP understanding.",
        "oracle_upper_bound_known_from_native_exact":True,
        "root_episodes":rows,
    }
    outfile.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"CPU_BASELINES_COMPLETE","cases":len(rows),
        "aggregates":d["full_results"]},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
