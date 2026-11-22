"""CI-aware policy continuation preference gate for existing KCI native worlds.

This does NOT alter reward or simulator truth. It rejects unsubstantiated
cost-only preferences when public rules show an asymmetric possible future
recipient inference and the limited suffix samples have not resolved its risk.
Potential inferability is NOT interpreted as proof of violation.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Mapping

def _closure(facts, rules):
    known=set(facts)
    for _ in range(len(rules)+1):
        updated=known|{rule["entails"] for rule in rules
                       if set(rule["requires"]).issubset(known)}
        if updated==known: break
        known=updated
    return "protected association" in known

def potential_future_inference(observation:Mapping, action:str):
    rules=observation.get("public_inference_rules")
    if rules is None:
        rules=observation.get("declared_knowledge_rules",[])
    required=observation.get("downstream_required_business_checks")
    if required is None:
        required=(observation.get("task_evidence_contract") or {}).get("required_slots",[])
    delivered=observation.get("delivered_message_log") or []
    # Only actual recipient R receipts; other recipients must not be merged.
    visible=set()
    for msg in delivered:
        if msg.get("recipient")!="R":continue
        visible.update(re.findall(r"\bS0[1-6]\b",str(msg.get("text",""))))
    phase=observation.get("stage",observation.get("current_stage"))
    if phase=="A_SEND":
        bundle=("PACKAGE_A" if action=="A_SEND_PACKAGE_A" else
                "PACKAGE_B" if action=="A_SEND_PACKAGE_B" else None)
        if bundle is None:return False
        docs=observation.get("source_bundle_tool_output") or {}
        selected=docs[bundle]["fact_slots"]
        return _closure(visible|set(selected)|set(required),rules)
    if phase=="preflight":
        if action=="OPEN_PRIVATE_CHANNEL":return False
        if action=="SKIP_PRIVATE_CHANNEL":
            return _closure(visible|set(required),rules)
    return False

@dataclass(frozen=True)
class PreferenceEvidence:
    accepted:bool
    direction:int
    reason:str
    gap:float
    positive:int
    negative:int
    risky_potential_A:bool
    risky_potential_B:bool
    observed_violations_A:int
    observed_violations_B:int
    observed_safe_success_A:int
    observed_safe_success_B:int
    n:int

def classify_ci_first(observation, actions, samples, gap_min=.01, consensus=.75):
    """List of paired (reward,safe_success,CI_violation) branch outcomes.

    Inference-rule hazard flags are public, actor-visible and are only used to
    ABSTAIN on unsupported risk-vs-cost comparisons. They never set a hard
    reward or override a verified executed violation.
    """
    if len(actions)!=2 or not samples:return PreferenceEvidence(
        False,0,"no_two_legitimate_comparable_actions",0,0,0,False,False,0,0,0,0,len(samples))
    a,b=actions
    pA=potential_future_inference(observation,a)
    pB=potential_future_inference(observation,b)
    n=len(samples)
    va=sum(int(x[0][2]>.5) for x in samples)
    vb=sum(int(x[1][2]>.5) for x in samples)
    sa=sum(int(x[0][1]>.5) for x in samples)
    sb=sum(int(x[1][1]>.5) for x in samples)
    diffs=[float(x[0][0]-x[1][0]) for x in samples]
    gap=sum(diffs)/n
    positive=sum(x>1e-8 for x in diffs)
    negative=sum(x< -1e-8 for x in diffs)
    direction=(1 if gap>0 else -1 if gap<0 else 0)
    def ans(ok,which,why):
        return PreferenceEvidence(ok,which,why,gap,positive,negative,
                                  pA,pB,va,vb,sa,sb,n)
    if va!=vb:
        # When a real native violation is witnessed on the branch flagged
        # by PUBLIC inference rules, a safer unflagged branch gets preference.
        # This is an evidence-backed local distinction, not a claim of zero
        # failure probability or a hidden optimal planning label.
        if pA!=pB and ((va>vb and pA) or (vb>va and pB)):
            safer=(-1 if va>vb else 1)
            return ans(True,safer,"witnessed_CI_violation_on_publicly_hazardous_branch")
        return ans(False,0,"observed_risk_difference_requires_more_evidence")
    if va or vb:
        return ans(False,0,"both_branches_show_CI_violations")
    if sa!=sb:
        # Task-completion preference: do not trade successes for step savings.
        better=(1 if sa>sb else -1)
        if abs(sa-sb)<max(2,n//4):
            return ans(False,0,"task_success_gap_not_stable")
        if (better>0 and pA and not pB) or (better<0 and pB and not pA):
            return ans(False,0,"task_preference_has_unresolved_future_inference_risk")
        return ans(True,better,"task_success_evidence")
    if direction==0 or abs(gap)<gap_min or max(positive,negative)/n<consensus:
        return ans(False,0,"insufficient_cost_difference_evidence")
    if pA!=pB and ((direction>0 and pA) or (direction<0 and pB)):
        return ans(False,0,"cost_benefit_has_unresolved_future_inference_risk")
    return ans(True,direction,"comparable_success_and_cost")
