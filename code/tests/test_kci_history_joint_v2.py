"""V2 equal-cost licensed broker is an acceptable action (NOT a mistake)."""
from pathlib import Path
import sys,torch,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_history_joint_three_arm_train_v2 import (
 get_data,matched_root_schedule,CASES,SEEDS,STEPS,ARMS,symmetric_kl_logits,joint_loss,
 RAW,VERIFY,ABORT)
D=ROOT/"artifacts/20261124_joint_history_supervision_v2"
def test_non_unique_valid_risky_actions_across_three_arms():
    data=get_data(D/"TRAIN_FIVE_CONTEXTS.jsonl")
    assert len(data)==7 and sum(map(len,data.values()))==35
    for root,g in data.items():
        assert set(g)==set(CASES)
        for cond,r in g.items():
            assert r["split"]=="train"
            assert r["native_gold"]=="SET_VALUED_REPLACED_IN_V2"
            assert set(r["acceptable_native_actions"]).isdisjoint(r["nonpreferred_same_state_comparators"])
            assert {RAW,VERIFY,ABORT}<=set(r["available_native_actions"])
            assert len(r["available_native_actions"]) in (3,4)
        if root in ("P206","P207"):
            assert set(g["risky"]["acceptable_native_actions"])=={VERIFY,"SEND_RAW_BROKER"}
            assert g["risky"]["nonpreferred_same_state_comparators"]==[RAW]
        else:
            assert g["risky"]["acceptable_native_actions"]==[VERIFY]
def test_all_methods_same_root_and_condition_example_schedule():
    data=get_data(D/"TRAIN_FIVE_CONTEXTS.jsonl")
    for seed in SEEDS:
        schedule=matched_root_schedule(data,seed)
        assert len(schedule)==STEPS and set(schedule)==set(data)
        assert schedule==matched_root_schedule(data,seed)
def test_sym_kl_and_old_context_gap_gradients_safe():
    ds={k:torch.nn.Parameter(torch.tensor(-0.2 if k=="risky" else 0.2)) for k in CASES}
    good=torch.nn.Parameter(torch.tensor([0.2,0.1,-0.5,0.3]))
    scores={k:(good if k in ("safe","hold","equivalent") else torch.tensor([0.0,0.2,0.1,-0.1],requires_grad=True)) for k in CASES}
    loss,extra=joint_loss(torch.tensor(1.,requires_grad=True),ds,scores)
    assert abs(extra["hold"].item())<1e-7
    assert abs(extra["equivalent"].item())<1e-7
    assert loss.isfinite()
    loss.backward()
    assert ds["risky"].grad.item()>0
def test_preregistered_set_valued_three_arm_comparison():
    j=json.loads((D/"THREE_ARM_PRE_REGISTERED.json").read_text(encoding="utf8"))
    assert j["all_arms_same_set_valued_native_witness_preference"]
    assert j["old_v1_experiment_retired_BEFORE_dev_evaluation"]
    assert j["three_train_methods"]==list(ARMS)
    assert j["train_rows"]==35 and j["dev_rows"]==15
