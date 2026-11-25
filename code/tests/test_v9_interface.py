"""Exercise semantic action binding, typed loss routing, and quarantine on actual delivery rows."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kci.v9_interface import Deliveries


@pytest.fixture(scope="module")
def corpus():
    root=Path(__file__).resolve().parents[3]/"kci_supervision_alignment_20261123"
    core=root/"d_future_core_20261124/D_future_core_multisource_v2"
    if not core.is_dir():pytest.skip("Internal delivery not installed")
    return Deliveries(root/"deliverables/KCI_VALIDATED_V9_INTERNAL",core)


def test_core_distillation_binds_exact_same_render_for_policy_and_baseline(corpus):
    r=corpus.core_reference("proofwriter_local_attestation",purpose="reference_distillation")[0]
    prompt,actions=corpus.training_example(r,7,0)
    calls=[]
    logits=torch.zeros(len(actions),requires_grad=True)
    def score(p,a):
        calls.append((p,a))
        return logits
    loss=corpus.loss(r,purpose="reference_distillation",seed=7,view=0,score=score,baseline=score)
    loss.backward()
    pos,neg=corpus.api.action_targets(r,actions)
    assert calls==[(prompt,actions)]*2 and logits.grad[pos[0]]<0<logits.grad[neg[0]]
    assert logits.grad.count_nonzero()==2
    with pytest.raises(ValueError):
        corpus.loss(r,purpose="training",seed=7,view=0,score=score,baseline=score)


@pytest.mark.parametrize("tier,kind",[("TASK_TERMINAL_NOT_CI","masked_action_set"),
                                      ("STATIC_EVIDENCE_SELECTION","multilabel_evidence_selection")])
def test_separate_auxiliary_losses(corpus,tier,kind):
    row=corpus.load_tier(tier,purpose="training",objective=kind)[0]
    captured=[]
    def score(p,menu):
        x=torch.zeros(len(menu),requires_grad=True)
        captured.append(x)
        return x
    loss=corpus.loss(row,purpose="training",seed=1234,view=1,score=score)
    loss.backward()
    assert torch.isfinite(loss) and captured[0].grad.count_nonzero()>0


@pytest.mark.parametrize("tier",["HOLD_CONTROL_ONLY","REF_COST_OR_STOCHASTIC_DIAGNOSTIC"])
def test_control_and_diagnostic_excluded(corpus,tier):
    with pytest.raises(ValueError):
        corpus.load_tier(tier,purpose="training",objective="semantic_pair_dpo")
    with pytest.raises(ValueError):
        corpus.core_reference("all",purpose="reference_distillation")

