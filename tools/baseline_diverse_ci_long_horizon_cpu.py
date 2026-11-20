r"""Quantify whether the new irreversible privacy-preflight business branch
requires more than myopic step-level CI checks. Real native CPU execution.
All policies get the exact same public declared Horn rules. No LLM test.
"""
from __future__ import annotations
import sys,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,BASE_FACTS
from kci.diverse_ci_long_horizon_env import LongHorizonCIEnv
from pilot_diverse_ci_long_horizon import (
    simulate,find_safe,conditions,correct_route,
)

def main():
    out=ROOT/"artifacts"/"20261119_composition_signals"/"diverse_ci_long_horizon_v2"/"CPU_POLICY_BASELINES.json"
    if out.exists():raise FileExistsError(out)
    cases=[];summ=Counter()
    for task in task_catalog():
        for cond in conditions(task):
            env=LongHorizonCIEnv(task,"linked",cond)
            route=correct_route(task,cond)
            best=find_safe(env,route)
            assert best and best["state"].success
            if route in ("summary","report_unavailable"):
                for name in ("myopic_skip","myopic_always_open","role_conditioned_preflight","oracle"):
                    result=best["state"]
                    cases.append({"root":task.id,"condition":cond,"policy":name,
                                  "success":result.success,"violation":result.violation,
                                  "steps":result.steps,"result":result.generated_result})
                    summ[name+"_success"]+=int(result.success)
                    summ[name+"_n"]+=1
                continue
            options={
                "myopic_skip":"SKIP_PRIVATE_CHANNEL",
                "myopic_always_open":"OPEN_PRIVATE_CHANNEL",
                "role_conditioned_preflight":("SKIP_PRIVATE_CHANNEL" if
                    task.business=="document" and cond in ("internal","relay") else
                    "OPEN_PRIVATE_CHANNEL"),
            }
            for name,option in options.items():
                result,trace=simulate(env,route,preflight=option,greedy=True)
                cases.append({
                    "root":task.id,"condition":cond,"policy":name,
                    "chosen_preflight":option,
                    "success":result.success,"violation":result.violation,
                    "steps":result.steps,"result":result.generated_result,
                    "last_action":trace[-1] if trace else None,
                })
                summ[name+"_success"]+=int(result.success and not result.violation)
                summ[name+"_violation"]+=int(result.violation)
                summ[name+"_safe_incomplete"]+=int(not result.success and not result.violation)
                summ[name+"_n"]+=1
            result=best["state"]
            cases.append({
                "root":task.id,"condition":cond,"policy":"oracle",
                "chosen_preflight":best["preflight"],
                "protected_facts":best["privacy_facts"],
                "broker_facts":best["broker_facts"],
                "success":result.success,"violation":result.violation,
                "steps":result.steps,"result":result.generated_result,
            })
            summ["oracle_success"]+=int(result.success)
            summ["oracle_n"]+=1
    assert all(summ[k+"_n"]==63 for k in (
        "myopic_skip","myopic_always_open","role_conditioned_preflight","oracle"))
    assert summ["oracle_success"]==63
    data={
        "study":"Knowledge accumulation and irreversible privacy-path branch CPU baselines",
        "n_distinct_business_roots":21,
        "n_business_conditions":63,
        "source":"native synthetic simulator; public declared inference rules",
        "result":dict(summ),
        "case_records":cases,
        "note":[
            "Myopic here knows the precise current recipient facts and rules, but cannot reopen the private verification channel after it skipped preflight.",
            "Always-open pays a two-step channel cost that can exceed the internal recipient deadline.",
            "The role-conditioned preflight heuristic uses business condition, authorization and current rule closure; it is a strong hand-written controller, not a trained policy.",
            "The oracle enumerates bounded privacy choices; not a model baseline.",
            "All cases use a deterministic scripted fictional task and are not independent natural user traces.",
            "No GPU, LLM, or API call performed."
        ]
    }
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in data.items() if k!="case_records"},
                     ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
