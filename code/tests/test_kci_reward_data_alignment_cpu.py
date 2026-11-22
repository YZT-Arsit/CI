"""Regression tests for the factual reward-data mismatch seen in KCI A-R-B-C."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_exact_policy_continuation_cpu import environment_public,enumerate_decisions,FrozenPolicyValue
from run_kci_role_local_credit_cpu import load_base
from run_kci_downstream_residual_cpu import RepairedDownstream
torch.set_num_threads(2)

def test_scalar_Qpi_reward_prefers_riskier_A_in_real_native_case():
    row=next(w for w in GROUP["train"] if w["world"]=="DG01_M01__tradeoff__swap0")
    base,_=load_base(7)
    model=RepairedDownstream(base).eval()
    ck=ROOT/"artifacts"/"20261121_kci_native_repair_mc_bridge_cpu_v1"/"CHECKPOINTS"/"KCI_DPR_BCCE_ACCE_seed7.pt"
    state=torch.load(ck,map_location="cpu",weights_only=True)
    model.R.load_state_dict(state["R"])
    model.B.load_state_dict(state["B"])
    env=environment_public(row)
    s,audit=env.transition(env.initial(),"A_READ_VERIFIED_PACKAGES")
    assert audit["verdict"]=="VALID"
    q=FrozenPolicyValue(env,model)
    qa=q.q(s,"A_SEND_PACKAGE_A")
    qb=q.q(s,"A_SEND_PACKAGE_B")
    assert qa[0]>qb[0]+.004
    assert qa[2]>qb[2]+.010
    assert qa[1]<qb[1]-.010
    assert env.transition(s,"A_SEND_PACKAGE_A")[1]["verdict"]=="VALID"
    assert env.transition(s,"A_SEND_PACKAGE_B")[1]["verdict"]=="VALID"

def test_both_legal_release_can_favor_legitimate_raw():
    row=GROUP["train"][0]
    env=environment_public(row)
    positions,_=enumerate_decisions(env)
    base,_=load_base(7)
    q=FrozenPolicyValue(env,base)
    count=0
    for p in positions:
        if p["phase"]!="release":continue
        if set(p["verdicts"].values()) != {"VALID"}:continue
        raw=q.q(p["state"],"SEND_RAW_R")
        private=q.q(p["state"],"VERIFY_PRIVATE_R")
        if raw[0]>private[0]+.005 and raw[2]<=private[2]+1e-4:
            count+=1
    assert count>0
