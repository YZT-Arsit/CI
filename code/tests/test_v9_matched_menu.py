from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kci.v9_matched_menu import matched_views, matched_semantic_scores


def render(h,actions):
    return h+"\nExecutable actions:\n"+"\n".join(actions)


def test_history_changes_only_authorized_text_and_input_permutations_are_equivalent():
    actions=["z","a","m"]
    canonical,p=matched_views(render("delivered A",actions),actions)
    other=["m","z","a"]
    c,q=matched_views(render("delivered B",other),other)
    assert c==canonical==["a","m","z"]
    assert all(x.replace("delivered A","delivered B")==y for x,y in zip(p,q))
    assert p[0].splitlines()[2:]==p[1].splitlines()[2:][::-1]


def test_semantic_rebinding_and_gradient_reach_both_views():
    actions=["z","a","m"];pars=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    calls=[]
    def score(text,batch):
        assert batch==["a","m","z"]
        calls.append(text)
        order=text.splitlines()[2:]
        return pars*(1 if len(calls)==1 else 3)+torch.tensor([order.index(a) for a in batch])
    output=matched_semantic_scores(render("history",actions),actions,score)
    assert len(calls)==2 and output.tolist()==[7.,3.,5.]
    output[0].backward()
    assert pars.grad.tolist()==[0.,0.,2.]


def test_mismatched_native_action_binding_is_rejected():
    with pytest.raises(ValueError):matched_views(render("history",["x","y"]),["y","x"])


def test_core_loader_binds_two_view_gradients_to_the_same_semantic_pair():
    from kci.v9_interface import Deliveries
    root=Path(__file__).resolve().parents[3]/"kci_supervision_alignment_20261123"
    core=root/"d_future_core_20261124/D_future_core_multisource_v2"
    if not core.exists():pytest.skip("Internal V9 corpus unavailable")
    delivery=Deliveries(root/"deliverables/KCI_VALIDATED_V9_INTERNAL",core)
    row=delivery.core_reference("proofwriter_local_attestation",purpose="reference_distillation")[0]
    n=len(row["executable_actions"])
    observed=[]
    for outer_view in (0,1):
        values=torch.zeros(n,requires_grad=True)
        def score(text,actions):
            return matched_semantic_scores(text,actions,lambda t,a:values)
        result=delivery.loss(row,purpose="reference_distillation",seed=7,view=outer_view,
                             score=score,baseline=lambda t,a:torch.zeros(n))
        result.backward()
        observed.append(values.grad.clone())
    assert torch.equal(*observed) and int(torch.count_nonzero(observed[0]))==2

