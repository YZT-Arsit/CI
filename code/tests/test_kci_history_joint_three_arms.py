"""Matched historical supervised diagnostic contract, no GPU in tests."""
from pathlib import Path
import sys,torch,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_history_joint_three_arm_train import (
    get_data,matched_root_schedule,CASES,SEEDS,STEPS,ARMS,symmetric_kl_logits,joint_loss,delta,RAW,VERIFY,ABORT)
D=ROOT/"artifacts/20261124_joint_history_supervision_v1"
def test_three_arms_exact_identical_five_native_states_per_root():
    data=get_data(D/"TRAIN_FIVE_CONTEXTS.jsonl")
    assert set(data)=={"P201","P202","P203","P205","P206","P207","P208"}
    assert sum(map(len,data.values()))==35
    assert len({r["actor_observation_sha256"] for g in data.values() for r in g.values()})==35
    for root,group in data.items():
        assert set(group)==set(CASES)
        for cond,r in group.items():
            assert RAW in r["available_native_actions"] and VERIFY in r["available_native_actions"]
            assert ABORT in r["available_native_actions"]
            assert r["native_gold"]==(VERIFY if cond=="risky" else RAW)
            assert r["source_evidence_grade"]=="real_native_immediate_verdict_with_fixed_suffix_only"
        assert group["risky"]["native_immediate_verdict"][RAW]=="VIOLATION"
        assert group["safe"]["native_immediate_verdict"][RAW]=="VALID"
        assert group["authorized"]["native_immediate_verdict"][RAW]=="VALID"
def test_prior_history_contrast_and_full_hold_behavior_loss():
    ds={c:torch.tensor(-.1 if c=="risky" else .1,requires_grad=True) for c in CASES}
    z=torch.nn.Parameter(torch.tensor([0.1,-0.2,0.3,0.8]))
    samples={c:z if c in ("safe","hold","equivalent") else torch.tensor([0.,0.,0.,0.],requires_grad=True) for c in CASES}
    j,terms=joint_loss(torch.tensor(1.0,requires_grad=True),ds,samples)
    assert j.isfinite().item()
    assert terms["hold"].item()==0 and terms["equivalent"].item()==0
    assert symmetric_kl_logits(z,z).abs()<1e-7
    j.backward()
    assert ds["risky"].grad is not None
def test_root_schedule_same_for_three_arms_no_extra_data():
    data=get_data(D/"TRAIN_FIVE_CONTEXTS.jsonl")
    assert set(ARMS)=={"action_ce","same_state_dpo","joint_flip_hold_ce"}
    for seed in SEEDS:
        schedule=matched_root_schedule(data,seed)
        assert len(schedule)==STEPS and set(schedule)==set(data)
        assert schedule==matched_root_schedule(data,seed)
def test_registered_contract_did_not_use_dev_for_training():
    c=json.loads((D/"THREE_ARM_PRE_REGISTERED.json").read_text(encoding="utf8"))
    assert c["same_data_sha256"]
    assert c["three_train_methods"]==list(ARMS)
    assert c["train_rows"]==35 and c["dev_rows"]==15
    assert c["reject_C_if_not_stably_better_than_CE_on_unseen_roots"]
