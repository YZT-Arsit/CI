"""Bounded, explicitly qualified post-main audit for long-horizon history-induced
preflight preference flips. Fixed task, business condition, authorization,
tool/budget and observation point; vary only previously delivered history.

This is exact over a documented restricted continuation family: fixed read order
(a..f) and all 2^6 private-action subsets, plus broker cert subsets when enabled.
It is NOT the full unrestricted native Qstar or Qpi of a 4B/7B policy.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,BASE_FACTS,digest
from kci.diverse_ci_long_horizon_env import LongHorizonCIEnv
from pilot_diverse_ci_long_horizon import simulate,conditions,correct_route

def all_masks():
    for n in range(7):
        for sub in combinations(BASE_FACTS,n):
            yield sub

def best_fixed_order(env,route,action):
    assert action in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")
    if route in ("summary","report_unavailable"):return None
    best=(0.0,None)
    plans=0
    for mask in all_masks():
        if action=="OPEN_PRIVATE_CHANNEL":
            st,tr=simulate(env,route,protect=mask,preflight=action)
            plans+=1
            val=1.0-0.01*st.steps if st.success and not st.violation else -1. if st.violation else 0.
            if val>best[0]:best=(round(val,6),{"private":list(mask),"broker":[],"steps":st.steps})
        elif route=="broker":
            # No preflight -> private attestation forbidden, but intermediary M
            # can see raw, then forward a certified assertion to R.
            st,tr=simulate(env,route,preflight=action,broker_for=mask)
            plans+=1
            val=1.0-0.01*st.steps if st.success and not st.violation else -1. if st.violation else 0.
            if val>best[0]:best=(round(val,6),{"private":[],"broker":list(mask),"steps":st.steps})
        elif not mask:
            st,tr=simulate(env,route,preflight=action)
            plans+=1
            val=1.0-0.01*st.steps if st.success and not st.violation else -1. if st.violation else 0.
            if val>best[0]:best=(round(val,6),{"private":[],"broker":[],"steps":st.steps})
    return {"value":best[0],"witness":best[1],"n_continuation_candidates":plans}

def main():
    BASE=ROOT/"artifacts"/"20261119_composition_signals"
    out=BASE/"P0_HISTORY_LONGTERM_FLIP_AUDIT_AFTER_MAIN.json"
    if out.exists():raise FileExistsError(out)
    records=[];counter=Counter()
    for task in task_catalog():
        for condition in conditions(task):
            route=correct_route(task,condition)
            if route in ("summary","report_unavailable"):
                counter["preflight_not_applicable"]+=1
                continue
            a={}
            for history in ("linked","unrelated","hold_a","hold_b"):
                env=LongHorizonCIEnv(task,history,condition,False)
                a[history]={x:best_fixed_order(env,route,x)
                  for x in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")}
            def gap(h):
                return round(a[h]["OPEN_PRIVATE_CHANNEL"]["value"]-
                         a[h]["SKIP_PRIVATE_CHANNEL"]["value"],6)
            related=gap("linked")
            unrelated=gap("unrelated")
            opposite=related*unrelated<0
            eq_hold=gap("hold_a")==gap("hold_b")
            records.append({
                "task":task.id,"graph":task.graph,"business":task.business,
                "condition":condition,"branch_route":route,
                "linked_minus_skip":related,
                "unrelated_minus_skip":unrelated,
                "hold_a_minus_skip":gap("hold_a"),
                "hold_b_minus_skip":gap("hold_b"),
                "strict_opposite_longterm_preflight_preference":opposite,
                "hold_reward_gap_same":eq_hold,
                "best_outcomes":a,
            })
            counter["preflight_comparable_scenarios"]+=1
            counter["strict_history_flip"]+=int(opposite)
            counter["tie_or_neutral_related"]+=int(related==0)
            counter["tie_or_neutral_unrelated"]+=int(unrelated==0)
            counter["hold_consistency"]+=int(eq_hold)
    result={
        "status":"FINITE_RESTRICTED_POLICY_AUDIT_COMPLETE",
        "methods":"BOTH legal preflight actions; all 64 private masks fixed source order; broker raw-to-M and certificate-to-R where available",
        "scope_limits":[
            "Not full Qstar. Assumes fixed fact inspection order and a fixed correct business route.",
            "Not Qpi. Continuations are scripted native agents, not current language-model policy.",
            "The return is 1-0.01*native_steps for safe success, 0 for safe failure, -1 for violation; a safe ABORT outside action masks is therefore at least as good as all failures.",
            "Business, authorization, tool contracts, native budget fixed within each history comparison. Only prior R delivered message history varies.",
            "A zero strict flip here is evidence the CURRENT design lacks a demonstrated P0 flip, not an impossibility proof for all possible future task designs."
        ],
        "counts":dict(counter),
        "records":records,
    }
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"counts":result["counts"],
        "first_five":records[:5]},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
