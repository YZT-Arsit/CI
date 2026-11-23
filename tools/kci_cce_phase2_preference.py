"""CCE predeclared CI-first comparison over identical frozen-policy samples.

Evidence quality: paired stochastic estimate with exact one-sided sign-test.
A pair is strictly preferred only if its CI-first comparison passes threshold.
Native exact Q* is NEVER used to create these labels or affect sampling.
"""
from __future__ import annotations
from math import comb
from collections import Counter

SEEDS=(7,1234,2025,42,314,2718,1618,99)
ALPHA=0.05
MIN_DISCORDANT=5
MIN_COST_COMPARABLE=5
TERMINAL_SCALAR_SUCCESS=1.0
TERMINAL_SCALAR_VIOLATION=-1.0
TERMINAL_SCALAR_SAFE_INCOMPLETE=0.0
TERMINAL_PREFERENCE_EPSILON=0.01

def matched_sign_test(left,right):
    assert len(left)==len(right) and len(left)>0
    n_plus=sum(a>b for a,b in zip(left,right))
    n_minus=sum(a<b for a,b in zip(left,right))
    n=n_plus+n_minus
    p=sum(comb(n,j) for j in range(max(n_plus,n_minus),n+1))/(2**n) if n else 1.
    accepted=n>=MIN_DISCORDANT and p<=ALPHA
    direction=("A" if n_plus>n_minus else "B" if n_minus>n_plus else None)
    return {"discordant":n,"A_greater":n_plus,"B_greater":n_minus,
            "ties":len(left)-n,"one_sided_exact_binomial_p":p,
            "direction_if_significant":direction if accepted else None}

def terminal_return(x):
    if x["ci_violation"]:return TERMINAL_SCALAR_VIOLATION
    if x["ci_safe_success"]:return TERMINAL_SCALAR_SUCCESS
    return TERMINAL_SCALAR_SAFE_INCOMPLETE

def compare(forks):
    """forks: one root/context with branches A_SEND_PACKAGE_A,B, paired seeds."""
    A=forks["A_SEND_PACKAGE_A"];B=forks["A_SEND_PACKAGE_B"]
    assert tuple(x["seed"] for x in A)==tuple(x["seed"] for x in B)==SEEDS
    for x in A+B:
        assert x["A_immediate_verdict"]=="VALID" and not x["A_immediate_CI_violation"]
    stats={}
    for key,rows in (("A",A),("B",B)):
        stats[key]={
          "n":len(rows),
          "unsafe_violation":sum(x["ci_violation"] for x in rows),
          "ci_safe_success":sum(x["ci_safe_success"] for x in rows),
          "safe_abort":sum(x["aborted"] and not x["ci_violation"] for x in rows),
          "deadline_missed":sum(x["deadline_missed"] for x in rows),
          "unknown_or_unfinished":sum(x["unknown_or_unfinished"] for x in rows),
          "mean_scalar_terminal":sum(map(terminal_return,rows))/len(rows),
          "successful_step_costs":[x["native_total_steps"] for x in rows if x["ci_safe_success"]]
        }
    scalar_delta=stats["A"]["mean_scalar_terminal"]-stats["B"]["mean_scalar_terminal"]
    terminal_action=("A_SEND_PACKAGE_A" if scalar_delta>TERMINAL_PREFERENCE_EPSILON
                     else "A_SEND_PACKAGE_B" if scalar_delta < -TERMINAL_PREFERENCE_EPSILON else None)
    result={
      "policy_evidence":"actual_frozen_4B_policy_pair_samples",
      "n_seed_matched_per_candidate":len(A),
      "ci_first_comparison_priority":["future_CI_violation","CI_safe_business_completion","native_success_cost"],
      "scalar_terminal_propagation":{
        "terminal_reward_contract":"1 if CI-safe task success, -1 if actual CI violation, 0 if safe incomplete (including abort/deadline)",
        "mean_A":stats["A"]["mean_scalar_terminal"],
        "mean_B":stats["B"]["mean_scalar_terminal"],
        "difference_A_minus_B":scalar_delta,
        "chosen_if_any":terminal_action,
        "no_uncertainty_filter":True},
      "branch_stats":stats,
      "preference":None,"abstain_reason":None,"evidence_layer":None,
      "tests":{}
    }
    if any(x["unknown_or_unfinished"] for x in A+B):
        result["abstain_reason"]="unknown_or_unfinished_future_policy_trace";return result
    # First compare future CI risk. Risk difference is not trumped by an
    # apparently better scalar terminal reward or lower execution cost.
    va=[int(not x["ci_violation"]) for x in A]
    vb=[int(not x["ci_violation"]) for x in B]
    v=matched_sign_test(va,vb);result["tests"]["CI_safety"]=v
    if v["direction_if_significant"]:
        label="A_SEND_PACKAGE_A" if v["direction_if_significant"]=="A" else "A_SEND_PACKAGE_B"
        result.update(preference=label,evidence_layer="paired_significant_future_CI_safety")
        return result
    if v["discordant"]:
        result["abstain_reason"]="unresolved_nonzero_future_CI_risk_difference";return result
    # Only equal observed safety can justify comparing task completion.
    sa=[int(x["ci_safe_success"]) for x in A]
    sb=[int(x["ci_safe_success"]) for x in B]
    s=matched_sign_test(sa,sb);result["tests"]["safe_task_success"]=s
    if s["direction_if_significant"]:
        result.update(preference=("A_SEND_PACKAGE_A" if s["direction_if_significant"]=="A" else "A_SEND_PACKAGE_B"),
                      evidence_layer="paired_significant_safe_task_success")
        return result
    if s["discordant"]:
        result["abstain_reason"]="uncertain_success_difference";return result
    # Cost only where paired trajectories BOTH achieved CI-safe completion.
    paired=[(a,b) for a,b in zip(A,B) if a["ci_safe_success"] and b["ci_safe_success"]]
    result["n_both_CI_safe_completions_for_cost"]=len(paired)
    if len(paired)<MIN_COST_COMPARABLE:
        result["abstain_reason"]="insufficient_pairwise_safe_success_for_cost";return result
    ca=[-a["native_total_steps"] for a,b in paired]
    cb=[-b["native_total_steps"] for a,b in paired]
    c=matched_sign_test(ca,cb);result["tests"]["cost_only_safe_completed"]=c
    if c["direction_if_significant"]:
        result.update(preference=("A_SEND_PACKAGE_A" if c["direction_if_significant"]=="A" else "A_SEND_PACKAGE_B"),
                      evidence_layer="paired_significant_cost_after_equal_CI_safe_success")
        return result
    result["abstain_reason"]="cost_tie_or_uncertain";return result
