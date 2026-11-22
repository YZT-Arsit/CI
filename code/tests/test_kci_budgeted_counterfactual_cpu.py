"""Regression tests for the budgeted sampled KCI counterfactual teacher."""
from __future__ import annotations
from pathlib import Path
import sys
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_exact_policy_continuation_cpu import FrozenPolicyValue
from run_kci_budgeted_counterfactual_cpu import (
    FrozenNativeSampler, fork_setup, load_policy, sample_pair_stream,
    classify,
)

torch.set_num_threads(2)

def test_pair_gate_direction_and_abstention():
    expected=[((.72,1.,0.),(.68,1.,0.)) for i in range(8)]
    v=classify(expected,4)
    assert v["accepted"] and v["direction"]==1
    same=[((.72,1.,0.),(.72,1.,0.)) for i in range(4)]
    assert not classify(same,4)["accepted"]
    mixed=[((1.,1.,0.),(0.,0.,0.)),((0.,0.,0.),(1.,1.,0.))]*2
    assert not classify(mixed,4)["accepted"]

def test_forks_are_native_real_and_marginal_probabilities_are_unchanged():
    row=GROUP["train"][0]
    env,st,children=fork_setup(row)
    assert len(children)==2
    assert not children[0].native.violation
    assert not children[1].native.violation
    actor=load_policy(7,"downstream_repaired")
    weights=[t.detach().clone() for t in actor.parameters()]
    sampler=FrozenNativeSampler(env,actor)
    q=FrozenPolicyValue(env,actor)
    exact=[q.q(st,a)[0] for a in ("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")]
    pairs=sample_pair_stream(env,children,sampler,40,seed=314159,paired=True)
    for i in (0,1):
        observed=float(np.mean([p[i][0] for p in pairs]))
        assert abs(observed-exact[i])<.18,(observed,exact[i])
    assert sampler.episodes==80
    assert sampler.native_steps>80
    assert all(torch.equal(x,y) for x,y in zip(weights,actor.parameters()))

def test_repeated_paired_counterfactual_is_reproducible():
    env,st,children=fork_setup(GROUP["train"][1])
    model=load_policy(7,"downstream_repaired")
    first=sample_pair_stream(env,children,FrozenNativeSampler(env,model),
                             10,seed=271828,paired=True)
    second=sample_pair_stream(env,children,FrozenNativeSampler(env,model),
                              10,seed=271828,paired=True)
    assert first==second
