"""Supervised root-contrast loss is sign-correct, not cross-history DPO."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_v4_history_contrast_train import context_gap_loss,MARGIN
def test_differentiable_context_gap_gradients():
    risk=torch.nn.Parameter(torch.tensor(0.4))
    safe=torch.nn.Parameter(torch.tensor(0.4))
    loss=context_gap_loss(risk,safe)
    loss.backward()
    assert loss.item()>0
    assert risk.grad.item()>0  # gradient descent lowers risk RAW odds
    assert safe.grad.item()<0  # gradient descent raises safe RAW odds
    assert MARGIN>0
def test_gap_reduction_when_context_discriminates():
    assert context_gap_loss(torch.tensor(-2.0),torch.tensor(2.0)).item() < context_gap_loss(torch.tensor(1.0),torch.tensor(1.0)).item()
