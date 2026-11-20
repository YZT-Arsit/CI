r"""CPU-only true long-horizon task branches with mandatory preflight decision.

A late one-step "avoid leakage now" rule cannot acquire private verification
after its registration deadline. The oracle searches bounded sets of protected
source messages, not a large model. No LLM, GPU or external API.
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.diverse_ci_branch_env import (
    task_catalog,BASE_FACTS,SLOT_LABELS,FINAL,closure,
)
from kci.diverse_ci_long_horizon_env import LongHorizonCIEnv

def conditions(t):
    if t.business=="calendar":return ("early_open","early_busy","both_busy")
    if t.business=="retail":return ("pending_yes","delivered_yes","pending_no")
    return ("external","relay","internal")

def correct_route(task,condition):
    if task.business=="calendar":
        return "late" if condition=="early_busy" else (
            "report_unavailable" if condition=="both_busy" else "early")
    if task.business=="retail":
        return "summary" if condition=="pending_no" else (
            "return" if condition=="delivered_yes" else "cancel")
    return "broker" if condition=="relay" else "direct"

def simulate(env,route,protect=(),preflight="OPEN_PRIVATE_CHANNEL",
             order=BASE_FACTS,broker_for=(),broker_forward_raw=(),
             greedy=False):
    s=env.initial()
    steps=[]
    def take(action):
        nonlocal s
        old=s
        if action not in env.options(s):
            steps.append({"action":action,"verdict":"NOT_OFFERED",
                          "step":s.steps})
            return False
        nxt,a=env.transition(s,action)
        steps.append({"action":action,"verdict":a["verdict"],
                      "step":nxt.steps,
                      "stage_after":nxt.stage})
        if a["verdict"] not in ("VALID","VIOLATION"):
            return False
        s=nxt
        return True
    assert take("INSPECT_BUSINESS")
    if route=="report_unavailable":
        assert take("REPORT_UNAVAILABLE")
        return s,steps
    assert take("CHOOSE:"+route),env.options(s)
    if s.done:return s,steps
    assert s.stage=="preflight"
    assert take(preflight)
    for f in order:
        if s.done:break
        if not take("READ:"+SLOT_LABELS[f]):break
        if s.done:break
        if greedy:
            # Greedy makes no early commitment, it responds to the imminent
            # knowledge increment. It cannot reopen the preflight after read.
            source=s.read_certs[f]["source"]["fact"]
            would_leak=(FINAL in closure((*s.delivered["R"],source),env.graph)
                        and not env._recipient_authorized(s))
            if would_leak:
                if "VERIFY_PRIVATE_R" in env.options(s):
                    action="VERIFY_PRIVATE_R"
                elif "SEND_RAW_M" in env.options(s):
                    action="SEND_RAW_M"
                else:
                    action="ABORT"
            else:action="SEND_RAW_R"
        elif f in broker_for:
            action="SEND_RAW_M"
        elif f in protect:
            action="VERIFY_PRIVATE_R"
        else:
            action="SEND_RAW_R"
        if not take(action):break
        if s.done:break
        if action=="VERIFY_PRIVATE_R":
            if not take("SEND_CERT_R"):break
        elif action=="SEND_RAW_M":
            if greedy:
                # Do not forward a current inference-triggering raw fact to R.
                action2=("FORWARD_RAW_R" if FINAL not in closure((
                    *s.delivered["R"],s.read_certs[f]["source"]["fact"]),env.graph)
                    or env._recipient_authorized(s) else "FORWARD_CERT_R")
            else:
                action2=("FORWARD_RAW_R" if f in broker_forward_raw else
                         "FORWARD_CERT_R")
            if not take(action2):break
    if not s.done and "COMMIT_RESULT" in env.options(s):
        take("COMMIT_RESULT")
    return s,steps

def find_safe(env,route):
    if route in ("report_unavailable","summary"):
        st,log=simulate(env,route,(),preflight="SKIP_PRIVATE_CHANNEL")
        return {"preflight":"NOT_NEEDED","privacy_facts":[],"broker_facts":[],
                "state":st,"trace":log} if st.success else None
    authorized=(env.task.business=="document" and env.condition=="internal")
    for prep in (("SKIP_PRIVATE_CHANNEL","OPEN_PRIVATE_CHANNEL")
                 if authorized else ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")):
        for n in range(7):
            for subset in combinations(BASE_FACTS,n):
                if prep=="SKIP_PRIVATE_CHANNEL" and subset:
                    if route!="broker":continue
                    # In broker route the intermediary may receive the raw
                    # document and send a safe certificate without opening
                    # the direct private verification channel.
                    st,log=simulate(env,route,(),preflight=prep,
                                    broker_for=subset)
                    if st.success and not st.violation:
                        return {"preflight":prep,"privacy_facts":[],
                                "broker_facts":list(subset),
                                "state":st,"trace":log}
                    continue
                st,log=simulate(env,route,subset,preflight=prep)
                if st.success and not st.violation:
                    return {"preflight":prep,"privacy_facts":list(subset),
                            "broker_facts":[],"state":st,"trace":log}
    return None

def main():
    stats=Counter();out=[]
    for task in task_catalog():
        for condition in conditions(task):
            env=LongHorizonCIEnv(task,"linked",condition)
            route=correct_route(task,condition)
            safe=find_safe(env,route)
            assert safe is not None and safe["state"].success,(task.id,condition)
            stats["worlds"]+=1;stats["safe_planner_success"]+=1
            if route in ("summary","report_unavailable"):
                out.append({"task":task.id,"condition":condition,
                            "route":route,"success":True,"preflight":"NOT_NEEDED"})
                continue
            greedy,greedylog=simulate(env,route,greedy=True,
                         preflight="SKIP_PRIVATE_CHANNEL")
            stats["myopic_skip_success"]+=int(greedy.success and not greedy.violation)
            stats["myopic_skip_violation"]+=int(greedy.violation)
            static,staticlog=simulate(env,route,preflight="OPEN_PRIVATE_CHANNEL",
                                      protect=safe["privacy_facts"],
                                      broker_for=safe["broker_facts"])
            stats["static_open_success_on_oracle_mask"]+=int(static.success and not static.violation)
            out.append({"task":task.id,"condition":condition,
                        "route":route,"planner_preflight":safe["preflight"],
                        "planner_private":safe["privacy_facts"],
                        "planner_broker":safe["broker_facts"],
                        "planner_success":safe["state"].success,
                        "myopic_skip_success":greedy.success,
                        "myopic_skip_final_stage":greedy.stage,
                        "static_open_success":static.success})
            print(task.id,task.graph,condition,route,
                  "planner",safe["preflight"],safe["privacy_facts"],safe["broker_facts"],
                  "greedy_success",greedy.success,
                  "always_open_success",static.success,flush=True)
    print(json.dumps({"stats":dict(stats),"records":len(out)},ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
