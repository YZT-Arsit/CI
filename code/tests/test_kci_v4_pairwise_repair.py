"""V4 grouped CE/margin supervision validation; no dev or final root read."""
from pathlib import Path
import sys,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_v4_pairwise_repair_train import group_from_verified_states,root_schedule,state_loss,WEIGHTS
D=ROOT/"artifacts/20261123_v4_semantic_control_v1/TRAIN_HISTORY_AWARE.jsonl"
def test_root_group_is_true_history_pair_and_authorized_same_state():
    roots=group_from_verified_states(D)
    assert len(roots)==3 and sorted(roots)==["DG31","DG32","DG33"]
    assert len({x["comparison_id"] for g in roots.values() for x in g.values()})==9
    for root,group in roots.items():
        assert set(group)=={"risky","safe","authorized"}
        assert group["risky"]["chosen_action"]=="VERIFY_PRIVATE_R"
        assert group["safe"]["chosen_action"]=="SEND_RAW_R"
        assert group["authorized"]["chosen_action"]=="SEND_RAW_R"
        assert group["risky"]["native_state_hash"]==group["authorized"]["native_state_hash"]
        assert group["risky"]["native_state_hash"]!=group["safe"]["native_state_hash"]
def test_96_root_updates_exactly_balanced_and_seed_stable():
    a=root_schedule(["DG31","DG32","DG33"],7,96)
    b=root_schedule(["DG31","DG32","DG33"],7,96)
    assert a==b and all(a.count(x)==32 for x in ("DG31","DG32","DG33"))
    assert sum(WEIGHTS.values())==1
def test_grouped_CE_and_margin_supervision_gradients_point_toward_opposite_actions():
    actions=("ABORT","SEND_RAW_R","VERIFY_PRIVATE_R")
    x=torch.nn.Parameter(torch.tensor([0.2,0.2,0.2]))
    for method in ("grouped_ce","margin_ce"):
        for cond,gold,target in (("risky","VERIFY_PRIVATE_R",-1.),("safe","SEND_RAW_R",1.),
                                 ("authorized","SEND_RAW_R",1.)):
            if x.grad is not None:x.grad.zero_()
            loss,ce,aux,delta=state_loss(actions,x,gold,cond,method)
            loss.backward()
            assert torch.isfinite(loss)
            assert x.grad[actions.index("SEND_RAW_R")]*target<0
            assert x.grad[actions.index("VERIFY_PRIVATE_R")]*target>0
