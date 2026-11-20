r"""Real simulator CPU baselines on the 7x3 knowledge/business pilot.

Policy baseline is deliberately granted the public declared Horn rules and
actually delivered R history. Its performance is a STRONG upper baseline
for template-like symbolic reasoning, not an LLM result.
"""
from pathlib import Path
from collections import Counter
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import *
from pilot_diverse_ci_branches import play,eligible_route,find_safe

def greedy(env,route):
    st=env.initial();log=[]
    def take(act):
        nonlocal st
        nxt,a=env.transition(st,act)
        log.append({"action":act,"verdict":a["verdict"]})
        if a["verdict"] not in ("VALID","VIOLATION"):
            raise RuntimeError("Unexpected invalid step: "+str((act,a)))
        st=nxt
    take("INSPECT_BUSINESS")
    take("CHOOSE:"+route)
    for fact in BASE_FACTS:
        if st.done:break
        take("READ:"+SLOT_LABELS[fact])
        source=st.read_certs[fact]["source"]
        private=(
            FINAL in closure((*st.delivered["R"],source["fact"]),env.graph)
            and not env._recipient_authorized(st)
        )
        if private:
            take("VERIFY_PRIVATE_R")
            if not st.done:take("SEND_CERT_R")
        else:
            take("SEND_RAW_R")
    if "COMMIT_RESULT" in env.options(st):
        take("COMMIT_RESULT")
    return st,log

def main():
    stats=Counter()
    rows=[]
    for t in task_catalog():
        conds=(
            ("early_open","early_busy") if t.business=="calendar" else
            ("pending_yes","delivered_yes","pending_no") if t.business=="retail" else
            ("internal","external","relay"))
        for condition in conds:
            e=DiverseCIEnv(t,"linked",condition)
            route=eligible_route(t,condition)
            for policy in ("raw","private_all","fixed_mask_a","myopic_symbolic","exact_search"):
                if policy=="raw":
                    st,trace=play(e,route)
                elif policy=="private_all":
                    st,trace=play(e,route,BASE_FACTS)
                elif policy=="fixed_mask_a":
                    st,trace=play(e,route,("a",))
                elif policy=="myopic_symbolic":
                    st,trace=greedy(e,route)
                else:
                    w=find_safe(e,route)
                    assert w
                    st=w["state"];trace=w["trace"]
                stats[policy+"_n"]+=1
                stats[policy+"_success"]+=int(st.success and not st.violation)
                stats[policy+"_violation"]+=int(st.violation)
                stats[policy+"_safe_incomplete"]+=int(not st.success and not st.violation)
                rows.append({
                    "task_id":t.id,"graph":t.graph,"business":t.business,
                    "condition":condition,"policy":policy,
                    "success":st.success,"violation":st.violation,
                    "steps":st.steps,"trace":trace})
    report={
        "status":"REAL_CPU_BASELINE_REPLAY",
        "n_world_conditions":sum(1 for r in rows if r["policy"]=="raw"),
        "stats":dict(stats),
        "rows":rows,
        "notes":[
            "The myopic policy has perfect access to the public rule graph and actually delivered receiver facts. It is a privileged symbolic competitor, not an LLM.",
            "The exact search enumerates all privacy-action subsets under a fixed business branch. It is an environment oracle.",
            "No model training, no external data, and no API used."
        ]
    }
    out=ROOT/"artifacts"/"20261119_composition_signals"/"diverse_ci_branch_v1"/"CPU_BASELINES.json"
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
