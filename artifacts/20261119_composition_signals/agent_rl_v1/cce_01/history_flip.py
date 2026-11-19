"""Audit whether CCE labels change when only delivered recipient history changes.
TRAIN/DEV horizon grids, fixed source tasks. Exact continuation labels from the
trusted synthetic environment; NEVER inputs to the model at inference.
"""
from collections import Counter,defaultdict
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,scenarios
from kci.dynamic_tasks import canonical
from audit_signal import qvalue
HERE=Path(__file__).resolve().parent
def states_for(spec,maxdepth=5):
    e=spec.build();start=e.decision_start()
    frontier=[((),start)]
    out={}
    while frontier:
        path,s=frontier.pop()
        if s.done:continue
        if len(e.options(s))>=2:
            out[path]={"role":e.actor(s),"phase":e.observation(s)["phase"],
                        "q":{a:qvalue(e,s,a) for a in e.options(s)}}
        if len(path)>=maxdepth:continue
        for action in e.options(s):
            nxt,verdict=e.transition(s,action)
            if verdict["verdict"]=="VALID" and not nxt.done:
                frontier.append((path+(action,),nxt))
    return out
def main():
    out=HERE/"history_flip"
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    rows=[];cnt=Counter();byrole=defaultdict(Counter)
    for split in ("train","dev"):
        for kind in ("meeting","query"):
            for allow in (False,True):
                horizons=sorted(set(x.horizon for x in scenarios(split)))
                for h in horizons:
                    no=states_for(Scenario(kind,False,allow,h))
                    yes=states_for(Scenario(kind,True,allow,h))
                    for path in sorted(set(no)&set(yes)):
                        a,b=no[path],yes[path]
                        if set(a["q"])!=set(b["q"]):raise AssertionError("Action set changed with history")
                        changed=[k for k in a["q"] if a["q"][k]!=b["q"][k]]
                        delta_compliance=[k for k in a["q"] if a["q"][k]["valid"]!=b["q"][k]["valid"]]
                        delta_reachability=[k for k in a["q"] if a["q"][k]["reachable"]!=b["q"][k]["reachable"]]
                        both_legal_future=[k for k in changed if a["q"][k]["valid"] and b["q"][k]["valid"]]
                        r={"split":split,"kind":kind,"allow":allow,"horizon":h,
                            "phase":a["phase"],"role":a["role"],"prefix":list(path),
                            "changed_actions":changed,"compliance_changed":delta_compliance,
                            "reachability_changed":delta_reachability,
                            "both_legal_future_change":both_legal_future,
                            "unrelated":a["q"],"related":b["q"]}
                        rows.append(r)
                        cnt["compared_prefixes"]+=1
                        if changed:cnt["history_sensitive_prefixes"]+=1
                        if delta_compliance:cnt["immediate_ci_change_prefixes"]+=1
                        if delta_reachability:cnt["future_reachability_change_prefixes"]+=1
                        if both_legal_future:cnt["both_legal_future_changed_prefixes"]+=1
                        byrole[a["role"]]["compared"]+=1
                        byrole[a["role"]]["history_sensitive"]+=bool(changed)
                        byrole[a["role"]]["reachability_changed"]+=bool(delta_reachability)
                        byrole[a["role"]]["both_legal_future_changed"]+=bool(both_legal_future)
    (out/"paired_prefixes.jsonl").write_text("".join(canonical(x)+"\n" for x in rows),encoding="utf-8")
    report={"checks":"PASS","source":"native source-adapted synthetic DynamicTask",
        "pairing":"same source+permission+action prefix, unrelated vs related delivered history",
        "max_prefix_actions":5,"counts":dict(cnt),"by_role":{k:dict(v) for k,v in byrole.items()},
        "no_gpu":True,"no_real_norm_validation":True,
        "cce_eligibility_does_not_prove_policy_expected_advantage":True}
    (out/"SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
