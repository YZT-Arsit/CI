r"""CPU search for true safe paths in seven new knowledge motifs x three
independent business control-flow families. No LLM, GPU, or external API.
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.diverse_ci_branch_env import *

def play(env,route,private=(),order=BASE_FACTS,broker_for=(),forward_raw=()):
    st=env.initial();trace=[]
    def take(action):
        nonlocal st
        nxt,a=env.transition(st,action)
        trace.append({"action":action,"verdict":a["verdict"],
                      "R_contains_secret":FINAL in env.knowledge(nxt,"R"),
                      "M_contains_secret":FINAL in env.knowledge(nxt,"M"),
                      "stage":nxt.stage,"step":nxt.steps})
        if a["verdict"] not in ("VALID","VIOLATION"):
            return False
        st=nxt
        return True
    assert take("INSPECT_BUSINESS")
    if "CHOOSE:"+route not in env.options(st):
        raise ValueError("Not a legal branch: "+route+" "+str(env.options(st)))
    assert take("CHOOSE:"+route)
    for f in order:
        if st.done:break
        if not take("READ:"+SLOT_LABELS[f]):break
        if f in private:
            if not take("VERIFY_PRIVATE_R"):break
            if not take("SEND_CERT_R"):break
        elif f in broker_for:
            if not take("SEND_RAW_M"):break
            if not take("FORWARD_RAW_R" if f in forward_raw else "FORWARD_CERT_R"):break
        else:
            if not take("SEND_RAW_R"):break
    if not st.done and "COMMIT_RESULT" in env.options(st):
        take("COMMIT_RESULT")
    return st,trace

def eligible_route(task,condition):
    if task.business=="calendar":
        return "late" if condition=="early_busy" else "early"
    if task.business=="retail":
        return "return" if condition=="delivered_yes" else (
            "summary" if condition=="pending_no" else "cancel")
    return "broker" if condition=="relay" else "direct"

def find_safe(env,route,order=BASE_FACTS,prefer_broker=False):
    for n in range(7):
        for subset in combinations(BASE_FACTS,n):
            if n>3:continue # budget makes any >3 impossible in this direct path
            st,t=play(env,route,subset,order)
            if st.success and not st.violation:
                return {"private":list(subset),"state":st,"trace":t}
    return None

def main():
    trials=Counter()
    proof_stats=[]
    for graph in GRAPH_LIBRARY:
        proofs=all_minimal_proofs(graph)
        assert proofs
        prior=witness_history(graph)
        assert FINAL not in closure(prior,GRAPH_LIBRARY[graph])
        assert FINAL in closure((*prior,"a"),GRAPH_LIBRARY[graph])
        proof_stats.append({"name":graph,"min_proofs":[list(p) for p in proofs],
                            "witness_history":list(prior),
                            "signature":graph_hyperedge_signature(graph)})
    print("GRAPH SIGNATURES",json.dumps(proof_stats,ensure_ascii=False),flush=True)
    for task in task_catalog():
        for cond in (
            ("early_open","early_busy") if task.business=="calendar" else
            ("pending_yes","delivered_yes","pending_no") if task.business=="retail" else
            ("internal","external","relay")):
            env=DiverseCIEnv(task,"linked",condition=cond)
            route=eligible_route(task,cond)
            safe=find_safe(env,route)
            assert safe is not None,("No safe route",task,cond)
            assert safe["state"].business
            raw,raw_tr=play(env,route,())
            allpriv,pt=play(env,route,BASE_FACTS)
            assert not allpriv.success,("all-private unexpectedly succeeded",task,cond)
            trials["scenarios"]+=1
            trials["safe_full"]+=int(safe["state"].success)
            trials["raw_violation"]+=int(raw.violation)
            trials["all_private_incomplete"]+=int(not allpriv.success)
            if cond in ("early_busy","delivered_yes","pending_no","relay"):
                trials["branch_status_cases"]+=1
            print(task.id,task.graph,task.business,cond,"route",route,
                  "private",safe["private"],"success",safe["state"].success,
                  "raw_viol",raw.violation,flush=True)
    print("SUMMARY",json.dumps(dict(trials),ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
