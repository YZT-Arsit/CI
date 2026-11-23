"""CI-first adaptive CCE statistical contract, seeded paired repeats."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools")]
from kci_cce_adaptive_protocol import (
    SEEDS32,FIRST_SEEDS,ALPHA_CONFIRM,paired_sign,wilson95,
    stage_signal,summarize_pair,should_continue,ACTION_A,ACTION_B,MIN_BOTH_SAFE_FOR_COST)
def row(seed,safe=False,viol=False,abort=False,deadline=False,steps=13):
    return {"seed":seed,"A_immediate_verdict":"VALID","A_immediate_CI_violation":False,
            "ci_violation":viol,"ci_safe_success":safe,"aborted":abort,"deadline_missed":deadline,
            "unknown_or_unfinished":False,"native_total_steps":steps}
def make(n,ab,bb):
    return [row(i,**ab) for i in SEEDS32[:n]],[row(i,**bb) for i in SEEDS32[:n]]
def test_disjoint_predeclared_seeds_and_fwer():
    assert len(SEEDS32)==len(set(SEEDS32))==32
    assert SEEDS32[:8]==FIRST_SEEDS
    assert ALPHA_CONFIRM==0.05/(14*2)
def test_accepted_FWER_confirm_requires_separate_blocks():
    a,b=make(16,{"safe":True},{"abort":True})
    v=stage_signal(a,b,alpha=ALPHA_CONFIRM,allow_confirm=True,seed_block_check=True)
    assert v["decision"]==ACTION_A and v["confirmed"]
    assert v["dimension"]=="CI_safe_completion"
    assert v["paired_sign"]["exact_two_sided_p"] < ALPHA_CONFIRM
    assert not should_continue(v,16)[0]
def test_future_CI_observed_once_abstains_without_cost_or_success_conflict():
    a,b=make(16,{"safe":True,"steps":11},{"safe":True,"steps":13})
    a[0]=row(SEEDS32[0],viol=True,steps=11)
    v=stage_signal(a,b,alpha=ALPHA_CONFIRM,allow_confirm=True,seed_block_check=True)
    assert v["status"]=="ABSTAIN_FEW_DISCORDANT_TRAJECTORIES"
    assert v["dimension"]=="CI_safety" and v["decision"] is None
def test_cost_preference_only_after_both_completed_safe():
    a,b=make(16,{"safe":True,"steps":12},{"safe":True,"steps":14})
    v=stage_signal(a,b,alpha=ALPHA_CONFIRM,allow_confirm=True,seed_block_check=True)
    assert v["decision"]==ACTION_A and v["dimension"]=="cost_given_both_safe_success"
    assert v["joint_safe_cost_sample_count"]==16
def test_K8_futility_does_not_claim_preference():
    a,b=make(8,{"abort":True},{"abort":True})
    s=summarize_pair(a,b,8)
    assert not s["K8_screen"]["confirmed"]
    assert s["current_look"]["decision"] is None
    go,reason=should_continue(s["current_look"],8)
    assert not go and "FUTILITY" in reason
def test_zero_future_violation_does_not_prove_zero_risk():
    assert wilson95(0,8)[1]>0.2
    assert wilson95(0,32)[1]>0.05
