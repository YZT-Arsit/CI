"""Tests for the unchanged-KCI native-to-text and CI-first pair interfaces."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_exact_policy_continuation_cpu import environment_public
from kci_llm_text_surface import render_role_text,decode_letter
from kci_ci_first_preference_gate import potential_future_inference,classify_ci_first

def _a_state():
    w=next(row for row in GROUP["train"] if row["world"]=="DG01_M01__tradeoff__swap0")
    e=environment_public(w)
    st, audit=e.transition(e.initial(),"A_READ_VERIFIED_PACKAGES")
    assert audit["verdict"]=="VALID"
    return e,st

def test_natural_language_exposes_authorized_a_and_b_observations_only():
    e,st=_a_state()
    obs=e.observation(st)
    prompt=render_role_text(obs)
    assert "scheduling evidence excerpt" in prompt
    assert "S01" not in prompt
    assert "A. Deliver signed evidence bundle A" in prompt
    assert "recipient" in prompt.lower()
    assert "Qstar" not in prompt and "native_state" not in prompt
    assert decode_letter(obs,"a")=="A_SEND_PACKAGE_A"
    st,_=e.transition(st,"A_SEND_PACKAGE_A")
    assert st.native.delivered["R"]
    b=e.observation(st)
    prompt_B=render_role_text(b)
    assert "Delivered receipt" in prompt_B
    assert "RecordsAgent" in prompt_B
    assert "source S0" not in prompt_B
    invalid=deepcopy(obs);invalid["success"]=True
    try:render_role_text(invalid)
    except ValueError:pass
    else:raise AssertionError("Hidden evaluator label exposed to model")

def test_public_inference_only_flags_possible_future_risk():
    _,st=_a_state()
    e,_=_a_state()
    obs=e.observation(st)
    assert potential_future_inference(obs,"A_SEND_PACKAGE_A")
    assert not potential_future_inference(obs,"A_SEND_PACKAGE_B")

def test_sparse_cost_signal_cannot_override_unresolved_ci_hazard():
    e,st=_a_state()
    obs=e.observation(st)
    acts=("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")
    # 4 samples miss a rare CI violation despite higher true future risk.
    samples=[((.68,1.,0.),(.66,1.,0.))]*4
    found=classify_ci_first(obs,acts,samples)
    assert not found.accepted
    assert found.reason=="cost_benefit_has_unresolved_future_inference_risk"

def test_verified_unsafe_witness_provides_safe_preference_without_qstar():
    e,st=_a_state()
    obs=e.observation(st)
    acts=("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")
    samples=[((-1.,0.,1.),(.664,1.,0.))]+[((.68,1.,0.),(.664,1.,0.))]*3
    found=classify_ci_first(obs,acts,samples)
    assert found.accepted and found.direction==-1
    assert found.reason=="witnessed_CI_violation_on_publicly_hazardous_branch"
