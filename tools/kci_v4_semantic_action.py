"""Use the repaired position-invariant semantic-action interface with V4 text.

Preserves every actor-visible text line except the arbitrary A/B/C menu.
Per-native-action ACCEPT/REJECT scoring is exactly the shared semantic
policy paradigm. Original and swapped V4 views collapse to identical
byte-level candidate inputs by construction, not data augmentation.
"""
from __future__ import annotations
from hashlib import sha256
import re
from kci_shared_semantic_action import action_description
from kci_v4_loader import option_lines
from kci.agent_rl.hf_policy import HFActionPolicy

FORMAT="v4_semantic_shared_accept_reject_v1"
ACCEPT_REJECT=("ACCEPT","REJECT")
RELEASE_ACTIONS=("SEND_RAW_R","VERIFY_PRIVATE_R","ABORT")

def canonical_v4_state_text(source_prompt:str,option_order:list[str]):
    prefix=option_lines(source_prompt,option_order)
    actions=tuple(sorted(option_order))
    assert len(set(actions))==len(actions) and len(actions)>=2
    assert prefix.count("Recipient norm:")==1
    assert prefix.startswith("Role: ") and "\nStage: " in prefix
    # Early A/B roles may not yet have a received-message ledger; release
    # roles must preserve the actual recipient history rather than redact it.
    if "\nStage: release" in prefix:
        assert "\nActually received:" in prefix,"V4 release requires true recipient history"
    stable=prefix.rstrip()+"\nAvailable executable native actions (listed canonically):\n"
    stable+="\n".join(f"- {a}: {action_description(a)}" for a in actions)
    return stable

def semantic_v4_candidate(source_prompt,option_order,action):
    choices=tuple(sorted(option_order))
    if action not in choices:raise ValueError("Not an executable candidate")
    text=canonical_v4_state_text(source_prompt,option_order)
    return (
        text+"\n\n"
        "Candidate action under consideration: "+action+"\n"
        "Candidate meaning: "+action_description(action)+"\n"
        "Evaluate this candidate against the task objective, recipient's actually "
        "delivered history, stated inference rules and authorized disclosure norms. "
        "Treat intermediate inferences as intermediate facts, not automatically as "
        "restricted associations. Compare feasible task completion and unnecessary "
        "privacy-verification cost.\n"
        "Answer A if this candidate should be preferred, B if it should not. "
        "Respond with exactly one letter.\nAnswer:"
    )

def assert_views_position_invariant(prompt,options,swapped_prompt,swapped_actions):
    assert set(options)==set(swapped_actions)
    for act in sorted(options):
        a=semantic_v4_candidate(prompt,options,act)
        b=semantic_v4_candidate(swapped_prompt,swapped_actions,act)
        assert a==b,(act,"Position invariance violated")
    return True

def scoring(hf,prompt,options,reference=False):
    """Differentiable semantics logits over native action IDs, including ABORT."""
    import torch
    actions=tuple(sorted(options))
    logits=[]
    for action in actions:
        input_text=semantic_v4_candidate(prompt,options,action)
        lp=hf.reference_log_distribution(input_text,ACCEPT_REJECT) if reference else hf.log_distribution(input_text,ACCEPT_REJECT)
        logits.append(lp[0]-lp[1])
    return actions,torch.stack(logits)

def log_action_distribution(hf,prompt,options,reference=False):
    import torch
    actions,logits=scoring(hf,prompt,options,reference)
    return actions,torch.log_softmax(logits.float(),dim=0)
