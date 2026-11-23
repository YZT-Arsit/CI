"""Single-adapter semantic policy: mathematical and prompt invariance tests."""
from copy import deepcopy
from pathlib import Path
import sys,json
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_shared_semantic_action import (canonical_actions, state_text,
                                        candidate_prompt, action_pair_loss)
from kci.diverse_ci_branch_env import digest
def sample():
    f=ROOT/"artifacts/20261123_shared_semantic_action_v1/V31_TRAIN_72.jsonl"
    return json.loads(f.read_text(encoding="utf8").splitlines()[0])["observation"]
def test_prompt_exactly_order_invariant_under_legal_action_shuffle():
    obs=sample()
    choices=list(obs["legal_actions"])
    prompt={a:candidate_prompt(obs,a) for a in choices}
    for order in (list(reversed(choices)), choices[1:]+choices[:1], sorted(choices)):
        other=deepcopy(obs);other["legal_actions"]=order
        assert state_text(other)==state_text(obs)
        assert all(candidate_prompt(other,a)==prompt[a] for a in choices)
def test_historical_changes_are_preserved_and_no_hidden_labels():
    obs=sample()
    obs2=deepcopy(obs)
    obs2["delivered_message_log"][-1]["text"]="Certified archived task evidence for case "+obs["current_case_subject"]+": source S06 was delivered to reviewer R."
    assert state_text(obs)!=state_text(obs2)
    assert "Native Qstar" not in state_text(obs)
    assert "Candidate action under consideration: OPEN_PRIVATE_CHANNEL" in candidate_prompt(obs,"OPEN_PRIVATE_CHANNEL")
    try:candidate_prompt(obs,"ILLEGAL_ORACLE_ACTION")
    except ValueError:pass
    else:raise AssertionError("Must reject hidden candidate")
def test_same_semantic_loss_under_action_permutation():
    obs=sample()
    actions=list(reversed(obs["legal_actions"]))
    vals={a:float(i)/10 for i,a in enumerate(sorted(actions))}
    scores=torch.tensor([vals[a] for a in sorted(actions)],requires_grad=True)
    l1=action_pair_loss(scores,actions,"OPEN_PRIVATE_CHANNEL",torch)
    l2=action_pair_loss(scores,list(reversed(actions)),"OPEN_PRIVATE_CHANNEL",torch)
    assert torch.allclose(l1,l2)
    l1.backward()
    assert scores.grad.abs().sum()>0
def test_source_release_uses_same_interface_and_corrected_rules():
    p=ROOT/"artifacts/20261123_shared_semantic_action_v1/RELEASE_TRAIN_STRICT8.jsonl"
    obs=json.loads(p.read_text(encoding="utf8").splitlines()[0])["observation"]
    assert obs["current_stage"]=="release"
    before={a:candidate_prompt(obs,a) for a in obs["legal_actions"]}
    changed=deepcopy(obs);changed["legal_actions"].reverse()
    assert all(candidate_prompt(changed,a)==before[a] for a in before)
    assert "Delivered receipt" in state_text(obs)
def test_shared_multi_role_same_policy_contract():
    p=ROOT/"artifacts/20261123_shared_semantic_action_v1/MULTIROLE_TRAIN.jsonl"
    data=[json.loads(x) for x in p.read_text(encoding="utf8").splitlines()]
    assert len(set(x["role"] for x in data))>=5
    for r in (data[0],data[100],data[-1]):
        assert candidate_prompt(r["observation"],r["chosen_action"])
        assert canonical_actions(r["observation"])==tuple(sorted(r["observation"]["legal_actions"]))
