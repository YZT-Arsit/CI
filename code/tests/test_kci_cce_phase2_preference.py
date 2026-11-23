"""CCE CI-first preference is evidence-gated, never Qstar-derived."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools")]
from kci_cce_phase2_preference import SEEDS,compare,matched_sign_test
def row(seed,*,safe=False,viol=False,steps=13,abort=False):
    return {"seed":seed,"A_immediate_verdict":"VALID","A_immediate_CI_violation":False,
        "ci_violation":viol,"ci_safe_success":safe,"aborted":abort,
        "deadline_missed":False,"unknown_or_unfinished":False,"native_total_steps":steps}
def test_valid_future_success_ranking_and_terminal_consistent():
    safe=[row(i,safe=True) for i in SEEDS];abort=[row(i,abort=True) for i in SEEDS]
    r=compare({"A_SEND_PACKAGE_A":safe,"A_SEND_PACKAGE_B":abort})
    assert r["preference"]=="A_SEND_PACKAGE_A"
    assert r["evidence_layer"]=="paired_significant_safe_task_success"
    assert r["scalar_terminal_propagation"]["chosen_if_any"]=="A_SEND_PACKAGE_A"
def test_one_rare_CI_violation_cannot_be_overridden_by_cost_or_task():
    a=[row(i,safe=True,steps=11) for i in SEEDS]
    a[0]=row(SEEDS[0],viol=True,steps=11)
    b=[row(i,safe=(k<3),abort=(k>=3),steps=15) for k,i in enumerate(SEEDS)]
    r=compare({"A_SEND_PACKAGE_A":a,"A_SEND_PACKAGE_B":b})
    assert r["preference"] is None
    assert r["abstain_reason"]=="unresolved_nonzero_future_CI_risk_difference"
    assert r["scalar_terminal_propagation"]["chosen_if_any"]=="A_SEND_PACKAGE_A"
def test_eight_consistent_actual_future_violation_supports_CI_first():
    bad=[row(i,viol=True) for i in SEEDS];good=[row(i,safe=True) for i in SEEDS]
    r=compare({"A_SEND_PACKAGE_A":bad,"A_SEND_PACKAGE_B":good})
    assert r["preference"]=="A_SEND_PACKAGE_B"
    assert r["evidence_layer"]=="paired_significant_future_CI_safety"
def test_cost_only_requires_comparable_joint_safe_success():
    a=[row(i,safe=True,steps=11) for i in SEEDS]
    b=[row(i,safe=True,steps=13) for i in SEEDS]
    r=compare({"A_SEND_PACKAGE_A":a,"A_SEND_PACKAGE_B":b})
    assert r["preference"]=="A_SEND_PACKAGE_A"
    assert r["evidence_layer"]=="paired_significant_cost_after_equal_CI_safe_success"
def test_all_safe_abort_yields_abstention_not_preference():
    a=[row(i,abort=True) for i in SEEDS];b=[row(i,abort=True) for i in SEEDS]
    r=compare({"A_SEND_PACKAGE_A":a,"A_SEND_PACKAGE_B":b})
    assert r["preference"] is None
    assert r["abstain_reason"]=="insufficient_pairwise_safe_success_for_cost"
