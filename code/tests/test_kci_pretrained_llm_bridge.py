"""Reproducible action-DPO bridge tests for actual frozen pretrained model traces."""
from __future__ import annotations
from pathlib import Path
import sys,json,hashlib
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_role_lora_dpo import validate_pairs,dpo_one,optimizer_step,MockRolePolicy
from kci_pretrained_llm_train_forks import compare
from kci_role_adapter_router import learning_role
from kci_llm_text_surface import render_role_text,decode_letter
from kci_exact_policy_continuation_cpu import environment_public
from compare_kci_chain_methods_cpu_action_interaction import GROUP

BASE=ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"
BP=BASE/"qwen7b_B_CCE_train8_v1"/"DPO_ROLE_B_TRAIN.jsonl"
RP=BASE/"source_R_verified_positive_v1"/"DPO_SOURCE_TRAIN.jsonl"

def test_real_pretrained_B_training_pairs_are_native_and_train_only():
    records=validate_pairs(BP,"Coordinator")
    assert len(records)==8
    assert len({r["root"] for r in records})==4
    assert all(r["chosen_action"]=="OPEN_PRIVATE_CHANNEL" and
               r["rejected_action"]=="SKIP_PRIVATE_CHANNEL" for r in records)
    for r in records:
        assert r["native_evidence"]["OPEN_PRIVATE_CHANNEL"]["safe_success"]
        assert not r["native_evidence"]["SKIP_PRIVATE_CHANNEL"]["safe_success"]
        assert "Deliver signed evidence bundle" not in r["prompt"] or \
               "private" in r["prompt"].lower()
        assert "Qstar" not in r["prompt"]
        assert r["sample_count_per_action"]==1

def test_real_R_training_pairs_include_legal_disclosures_and_native_repair():
    records=validate_pairs(RP,"SourceAgent")
    assert len(records)==100
    assert sum(r["chosen_action"]=="SEND_RAW_R" for r in records)==72
    assert sum(r["chosen_action"]=="VERIFY_PRIVATE_R" for r in records)==28
    assert len({r["root"] for r in records})==4
    assert all(r["stage"]=="release" for r in records)

def test_pretrained_data_never_imports_developer_seen_test_into_training():
    assert all(row["root"] in ("DG01","DG05","DG09","DG10")
               for row in validate_pairs(BP,"Coordinator"))
    assert all(row["root"] in ("DG01","DG05","DG09","DG10")
               for row in validate_pairs(RP,"SourceAgent"))
    assert all(row["root"] not in ("DG14","DG18","DG19")
               for row in validate_pairs(BP,"Coordinator"))

def test_dpo_update_changes_acting_role_action_log_odds():
    records=validate_pairs(BP,"Coordinator")
    actor=MockRolePolicy()
    optim=torch.optim.AdamW(actor.trainable_parameters(),lr=.08,weight_decay=0)
    before=float(actor.scores[0].detach()-actor.scores[1].detach())
    for i in range(20):
        result=optimizer_step(actor,[records[i%len(records)]],
                              optim,beta=2.0,grad_clip=1.)
        assert result["preclip_gradient_norm"]>0
    after=float(actor.scores[0].detach()-actor.scores[1].detach())
    assert after>before+.4

def test_native_llm_role_text_maps_action_letters_exactly():
    w=GROUP["train"][0]
    e=environment_public(w)
    state,_=e.transition(e.initial(),"A_READ_VERIFIED_PACKAGES")
    o=e.observation(state)
    prompt=render_role_text(o)
    assert "native_state" not in prompt
    assert "S01" not in prompt
    assert decode_letter(o,"A")==o["legal_actions"][0]
    with pytest.raises(ValueError):decode_letter(o,"OPEN_PRIVATE_CHANNEL")
    with pytest.raises(ValueError):decode_letter(o,"X")

def test_role_adapter_dispatch_uses_acting_native_decision_stage():
    assert learning_role({"stage":"A_SEND","actor":"RecordsAgent"})=="RecordsAgent"
    assert learning_role({"current_stage":"preflight","actor":"Coordinator"})=="Coordinator"
    assert learning_role({"current_stage":"release","actor":"DirectoryAgent"})=="SourceAgent"
    assert learning_role({"current_stage":"choose_fact","actor":"BusinessExecutor"})=="BusinessExecutor"
    assert learning_role({"current_stage":"check","actor":"Coordinator"})=="None"

def test_evidence_winner_comes_from_actual_native_completion():
    a={"terminal":"DONE","safe_success":True,"CI_violation":False,"reward":.6}
    b={"terminal":"DONE","safe_success":False,"CI_violation":False,"reward":0.0}
    sign,kind=compare(a,b)
    assert sign==1 and kind=="verified_task_success"
    sign,kind=compare(b,a)
    assert sign==-1 and kind=="verified_task_success"
