"""Mathematical contract of the shared semantic DPO implementation."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_shared_semantic_dpo import rejected,margin
def sample(name):
    path=ROOT/"artifacts"/"20261123_shared_semantic_action_v1"/name
    return json.loads(path.read_text(encoding="utf8").splitlines()[0])
def test_rejected_action_same_actor_state_and_feasible():
    for name in ("V31_TRAIN_72.jsonl","RELEASE_TRAIN_STRICT8.jsonl"):
        r=sample(name)
        assert rejected(r)!=r["chosen_action"]
        assert rejected(r) in r["observation"]["legal_actions"]
def test_shared_dpo_preference_margin_reference_shift():
    r=sample("RELEASE_TRAIN_STRICT8.jsonl")
    acts=tuple(sorted(r["observation"]["legal_actions"]))
    p=torch.nn.Parameter(torch.zeros(len(acts)))
    reference_advantage=0.7
    d=margin(p,acts,r)-reference_advantage
    loss=-torch.nn.functional.logsigmoid(2*d)
    loss.backward()
    assert p.grad[acts.index(r["chosen_action"])]<0
    assert p.grad[acts.index(rejected(r))]>0
def test_only_one_shared_adapter_architecture():
    text=(ROOT/"tools/kci_shared_semantic_dpo.py").read_text(encoding="utf8")
    assert "set_adapter" not in text and "load_adapter" not in text
    assert "lora_rank=8" in text
