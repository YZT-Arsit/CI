"""Restart real multi-agent KCI rollouts from a verified, authorized prefix.

The native DynamicTask state is ALWAYS reconstructed by replaying valid prior
actions, never by writing simulator state flags or injecting precomputed labels.
The policy sees only its acting role's canonical observation, NOT restart
selection metadata, Qstar, target action or future outcomes.

A restarted episode's initial_state_hash denotes the VERIFIED intermediate
native state. Terminal reward covers the complete task (including prefix cost).
Only actions sampled AFTER restart are policy-gradient trainable.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from pathlib import Path
import math,random
from typing import Any

from kci.dynamic_tasks import canonical
from .core import (ActionPolicy,Decision,Episode,LETTERS,Scenario,digest,
                   prompt_for,group_advantages,terminal_reward)
from .policy_continuation import FrozenFork,replay_prefix


@dataclass(frozen=True)
class RestartSpec:
    pair_id: int
    scenario: Scenario
    prefix: tuple[str,...]
    state_hash: str
    role: str
    phase: str
    selected_source: str = "archived_4b_TRAIN_prefix"

    @property
    def key(self):
        return self.scenario.key+":"+self.state_hash

    def to_fork(self):
        return FrozenFork(self.scenario,self.prefix,self.phase,self.role,self.state_hash)

    @classmethod
    def from_record(cls,raw):
        if type(raw["pair_id"]) is not int or raw["pair_id"]<0:
            raise ValueError("Invalid paired state ID")
        scenario=Scenario(**raw["scenario"])
        if type(raw["prefix"]) is not list or not all(type(x) is str for x in raw["prefix"]):
            raise ValueError("Prefix must be a list of exact action names")
        return cls(raw["pair_id"],scenario,tuple(raw["prefix"]),
                   raw["state_hash"],raw["role"],raw["phase"])

    def serialize(self):
        return {"pair_id":self.pair_id,"scenario":asdict(self.scenario),
                "prefix":list(self.prefix),"state_hash":self.state_hash,
                "role":self.role,"phase":self.phase,
                "selected_source":self.selected_source}


def restore(spec:RestartSpec,source_root=None):
    env,state=replay_prefix(spec.scenario,spec.prefix,source_root)
    if state.done or state.violation:
        raise RuntimeError("Cannot restart a terminal or unsafe native prefix")
    if digest(asdict(state))!=spec.state_hash:
        raise RuntimeError("Restart state hash differs from pinned native prefix")
    observation=env.observation(state)
    if env.actor(state)!=spec.role or observation["phase"]!=spec.phase:
        raise RuntimeError("Restart role/phase differs from pinned contract")
    return env,state


def execute_restart_episode(spec:RestartSpec,policy:ActionPolicy,
                            sample_index:int,seed:int,
                            *,split="train",source_root=None)->Episode:
    if sample_index<0 or type(seed) is not int:
        raise ValueError("Invalid restart rollout sample/seed")
    env,state=restore(spec,source_root)
    rng=random.Random(seed)
    decisions=[]
    while not state.done:
        actions=tuple(env.options(state))
        if not actions:raise RuntimeError("Nonterminal native prefix has no executable options")
        obs=env.observation(state)
        text=prompt_for(obs,actions)
        sample=policy.choose(text,actions,rng)
        if type(sample.index) is not int or not 0<=sample.index<len(actions):
            raise RuntimeError("Restart policy selected an invalid action")
        if not all(math.isfinite(x) for x in (sample.logprob,sample.entropy)):
            raise RuntimeError("Invalid model statistics")
        chosen=actions[sample.index]
        next_state,audit=env.transition(state,chosen)
        if audit["verdict"]=="UNKNOWN":
            raise RuntimeError("Unknown action semantics cannot be scored")
        if next_state.turn<=state.turn:
            raise RuntimeError("No progress from model action")
        decisions.append(Decision(
            actor=env.actor(state),state_hash=digest(asdict(state)),
            observation_hash=digest(obs),prompt=text,actions=list(actions),
            chosen_index=sample.index,chosen_action=chosen,
            chosen_letter=LETTERS[sample.index],chosen_token_id=sample.choice_token_id,
            old_logprob=sample.logprob,entropy=sample.entropy,
            verdict=audit["verdict"],
            newly_derived=list(audit.get("new_protected_relations",[])),
            receipt_count_before=len(state.ledger),
            receipt_count_after=len(next_state.ledger)))
        state=next_state
        if len(decisions)>env.horizon+1:
            raise RuntimeError("Restart rollout exceeded explicit horizon")
    if state.success and state.violation:
        raise AssertionError("Unsafe terminal marked successful")
    return Episode(scenario_key=spec.scenario.key,
       sample_index=sample_index,rollout_seed=seed,policy_id=policy.policy_id,
       split=split,initial_state_hash=spec.state_hash,
       decisions=decisions,success=bool(state.success),
       violation=bool(state.violation),done=bool(state.done),
       total_steps=state.turn,
       reward=terminal_reward(success=state.success,violation=state.violation,
                              done=state.done,steps=state.turn),
       final_receipt_digest=digest(state.ledger))


def sample_restart_group(spec:RestartSpec,policy:ActionPolicy,
                         n:int,seed:int,*,split="train",source_root=None):
    if not 2<=n<=32:raise ValueError("Group size must be 2..32")
    group=[execute_restart_episode(
        spec,policy,i,seed+10007*i,split=split,source_root=source_root)
        for i in range(n)]
    group_advantages(group)
    if any(not x.decisions or x.decisions[0].state_hash!=spec.state_hash for x in group):
        raise AssertionError("Restart group failed to use exactly the intended prefix")
    return group
