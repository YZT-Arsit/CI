"""Matched update multisets with different order only."""


def make_schedule(arm):
    correct=[['correct',i] for i in range(27)]
    replay=[['replay',i] for i in range(80)]
    if arm=='correct_then_replay':return correct+replay
    if arm=='replay_then_correct':return replay+correct
    assert arm=='interleaved'
    return [item for i in range(27) for item in replay[3*i:3*i+3]+[correct[i]]]
