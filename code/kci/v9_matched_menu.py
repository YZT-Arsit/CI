"""Two history-independent menu displays for a full semantic-reply likelihood scorer."""
from hashlib import sha256
import json


def matched_views(prompt, actions, seed=7):
    separator = "\nExecutable actions:\n"
    head, sep, tail = prompt.partition(separator)
    if not sep or tail.splitlines() != list(actions) or not actions or len(set(actions)) != len(actions):
        raise ValueError("Expected a nonempty native menu with unique, bound semantic actions")
    canonical = sorted(actions)
    def order_key(action):
        encoded = json.dumps([seed, action], ensure_ascii=False, separators=(",", ":")).encode("utf8")
        return sha256(encoded).digest()
    forward = sorted(canonical, key=order_key)
    return canonical, tuple(head + separator + "\n".join(display) for display in (forward, forward[::-1]))


def matched_semantic_scores(prompt, actions, score, seed=7):
    """Average both text views; keep internal scoring batch order fixed and rebind outputs.

    score receives actor-visible text and semantic completions in canonical batch order.
    Returned values follow the caller's original action order and keep both gradient paths.
    """
    canonical, views = matched_views(prompt, actions, seed)
    first, reverse = (score(text, canonical) for text in views)
    if first.shape != (len(actions),) or reverse.shape != first.shape:
        raise ValueError("Expected one semantic score per canonical action")
    averaged = (first + reverse) / 2
    index = {action: i for i, action in enumerate(canonical)}
    return averaged[[index[a] for a in actions]]

