"""Position-free semantic-action policy interface for one shared LLM adapter.

Given an actor-visible observation, every executable native action is scored
through the SAME yes/no head. No option letters or role-specific adapters are
used. The final probability distribution is over native semantic action IDs.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
from typing import Mapping

from kci_llm_text_surface import ACTION_NAMES, render_role_text, _replace_codes

FORMAT_VERSION = "shared_semantic_action_v1_yes_no"
ACCEPT = "A"
REJECT = "B"

def action_description(action: str) -> str:
    if action in ACTION_NAMES:
        return ACTION_NAMES[action]
    if action.startswith("READ:"):
        return "Read certified source record " + action.partition(":")[2]
    if action.startswith("CHOOSE:"):
        return "Select task route " + action.partition(":")[2]
    return action.replace("_", " ")

def canonical_actions(obs: Mapping) -> tuple[str, ...]:
    actions = obs["legal_actions"]
    if not isinstance(actions, list) or not actions or len(actions) > 26:
        raise ValueError("No executable native action menu")
    if len(set(actions)) != len(actions):
        raise ValueError("Duplicate executable native actions")
    return tuple(sorted(actions))

def state_text(obs: Mapping) -> str:
    # Keep every authorized state field from the existing audited text renderer.
    # Discard the letter-ordered menu, replace it with a canonical semantic menu.
    stable = deepcopy(dict(obs))
    actions = canonical_actions(stable)
    stable["legal_actions"] = list(actions)
    rendered = render_role_text(stable)
    header = "Executable actions (reply with one option letter):"
    prefix, delim, rest = rendered.partition(header)
    if not delim:
        raise ValueError("Text renderer no longer provides menu boundary")
    if prefix.count("Recipient privacy policy:") != 1:
        raise ValueError("Missing recipient policy")
    unordered = "Available executable native actions (listed canonically):\n"
    unordered += "\n".join(f"- {act}: {action_description(act)}" for act in actions)
    return prefix.rstrip() + "\n" + unordered

def candidate_prompt(obs: Mapping, action: str) -> str:
    actions = canonical_actions(obs)
    if action not in actions:
        raise ValueError("Candidate not executable under actor observation")
    return (
        state_text(obs) + "\n\n"
        "Candidate action under consideration: " + action + "\n"
        "Candidate meaning: " + action_description(action) + "\n"
        "Evaluate this candidate against the task objective, recipient's actually "
        "delivered history, stated inference rules and authorized disclosure norms. "
        "Treat intermediate inferences as intermediate facts, not automatically as "
        "restricted associations. Compare feasible task completion and unnecessary "
        "privacy-verification cost.\n"
        "Answer A if this candidate should be preferred, B if it should not. "
        "Respond with exactly one letter.\nAnswer:"
    )

def prompt_fingerprint(obs: Mapping, action: str) -> str:
    return sha256(candidate_prompt(obs, action).encode("utf-8")).hexdigest()

def action_pair_loss(scores, actions, chosen, torch):
    """Cross-entropy on semantic actions, never on an A/B menu position."""
    canonical = tuple(sorted(actions))
    if chosen not in canonical or len(set(canonical)) != len(canonical):
        raise ValueError("Invalid supervision")
    assert scores.numel() == len(canonical)
    ix = canonical.index(chosen)
    return -torch.log_softmax(scores, dim=0)[ix]

class SharedSemanticPolicy:
    """Exactly one LoRA adapter for every role (trainable or frozen).

    The same model is used for RecordsAgent, Coordinator and every downstream
    agent. All role differences reside ONLY in authorized observations.
    """
    def __init__(self, hf):
        self.hf = hf
        self._cache = {}

    def scores(self, obs):
        import torch
        actions = canonical_actions(obs)
        values = []
        for action in actions:
            prompt = candidate_prompt(obs, action)
            # A=prefer this candidate, B=do not prefer. Two tokens have
            # fixed semantic meaning, irrespective of action ordering.
            two = self.hf.log_distribution(prompt, ("ACCEPT", "REJECT"))
            values.append(two[0] - two[1])
        return actions, torch.stack(values)

    def distribution(self, obs):
        import torch
        actions, scores = self.scores(obs)
        prob = torch.softmax(scores.float(), dim=0)
        return actions, prob

    def decide(self, obs):
        import torch
        with torch.inference_mode():
            actions, prob = self.distribution(obs)
        idx = int(prob.argmax())
        return actions[idx], {a:float(p) for a,p in zip(actions,prob.cpu().tolist())}
