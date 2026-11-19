"""Frozen-policy counterfactual continuation at real KCI Agent decision states.

Q_pi(s,a) is estimated by executing each immediately valid candidate action,
then K native multi-agent trajectories sampled from the SAME frozen policy.
Only role-authorized observations are shown to the LM. Native exact Qstar is
separately recorded as a diagnostic, NEVER used for policy actions.
No model update is performed in this module.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import math,random,statistics
from typing import Any
from kci.dynamic_tasks import DynamicTask
from .core import ActionPolicy,Scenario,digest,prompt_for,terminal_reward
from .cce import EARLY,EXCLUDED

@dataclass(frozen=True)
class FrozenFork:
    scenario: Scenario
    prefix: tuple[str,...]
    phase: str
    role: str
    state_hash: str

    @property
    def key(self):
        return self.scenario.key+":"+self.state_hash

def replay_prefix(scenario:Scenario,prefix:tuple[str,...],source_root=None):
    env=scenario.build() if source_root is None else scenario.build(source_root)
    state=env.decision_start()
    for action in prefix:
        if state.done or action not in env.options(state):
            raise ValueError("Fork prefix is not executable")
        state,audit=env.transition(state,action)
        if audit["verdict"]!="VALID":
            raise ValueError("Fork prefix contains unsafe or UNKNOWN action")
    if state.done:
        raise ValueError("Cannot fork a terminal prefix")
    return env,state

def make_fork(scenario,prefix,source_root=None)->FrozenFork:
    env,state=replay_prefix(scenario,tuple(prefix),source_root)
    obs=env.observation(state)
    return FrozenFork(scenario=scenario,prefix=tuple(prefix),phase=obs["phase"],
                      role=env.actor(state),state_hash=digest(asdict(state)))

def immediately_legal(env,state,*,exclude=EXCLUDED):
    legal={}
    for action in env.options(state):
        if action in exclude:continue
        nextstate,audit=env.transition(state,action)
        if audit["verdict"]=="UNKNOWN":
            raise RuntimeError("Native action produced UNKNOWN")
        if audit["verdict"]=="VALID":
            legal[action]=nextstate
    return legal

def frozen_continuation(env:DynamicTask,state,action:str,
                        policy:ActionPolicy,seed:int)->dict:
    """Force one candidate, then follow only model-generated actions to terminal."""
    if action not in env.options(state):
        raise ValueError("Counterfactual action not offered in native state")
    initial_hash=digest(asdict(state))
    after,audit=env.transition(state,action)
    if audit["verdict"]=="UNKNOWN":
        raise RuntimeError("UNKNOWN transition cannot receive a reward")
    rng=random.Random(seed)
    branch=[action]
    generated_logprobs=[]
    current=after
    while not current.done:
        options=tuple(env.options(current))
        if not options:raise RuntimeError("No actions before terminal")
        policy_prompt=prompt_for(env.observation(current),options)
        sample=policy.choose(policy_prompt,options,rng)
        if not 0<=sample.index<len(options):
            raise RuntimeError("Frozen model chose invalid action")
        if not math.isfinite(sample.logprob):
            raise RuntimeError("Frozen model produced nonfinite logprob")
        chosen=options[sample.index]
        nxt,audit=env.transition(current,chosen)
        if audit["verdict"]=="UNKNOWN":
            raise RuntimeError("UNKNOWN during continuation")
        if nxt.turn<=current.turn:
            raise RuntimeError("Continuation did not advance")
        branch.append(chosen)
        generated_logprobs.append(sample.logprob)
        current=nxt
        if len(branch)>env.horizon+1:
            raise RuntimeError("Exceeded horizon")
    if current.success and current.violation:
        raise AssertionError("CI-unsafe result labeled success")
    return {
        "forced_action":action,"prefix_hash":initial_hash,
        "seed":seed,"policy_id":policy.policy_id,
        "reward":terminal_reward(success=current.success,violation=current.violation,
                                 done=current.done,steps=current.turn),
        "safe_success":bool(current.success and not current.violation),
        "ci_violation":bool(current.violation),
        "safe_failure":bool(not current.success and not current.violation),
        "final_steps":int(current.turn),
        "continuation_actions":branch,
        "continuation_logprob_count":len(generated_logprobs),
        "final_receipts_hash":digest(current.ledger),
    }

def frozen_policy_counterfactual(
        fork:FrozenFork,policy:ActionPolicy, *,
        n_continuations:int=4,seed:int=9111926,source_root=None,
        compute_oracle:bool=True,require_early:bool=False)->dict:
    """Compare all simultaneously legal non-abort actions under frozen pi.

    Common random-number seeds are shared across the alternatives.
    """
    if not 2<=n_continuations<=16:
        raise ValueError("Use 2..16 continuations per action")
    env,state=replay_prefix(fork.scenario,fork.prefix,source_root)
    if digest(asdict(state))!=fork.state_hash:
        raise RuntimeError("Fork replay state hash mismatch")
    if (env.actor(state),env.observation(state)["phase"])!=(fork.role,fork.phase):
        raise RuntimeError("Fork actor/phase mismatch")
    if require_early and (fork.role,fork.phase) not in EARLY:
        raise ValueError("Not an early CCE decision")
    options=immediately_legal(env,state)
    if len(options)<2:
        raise ValueError("Fewer than two immediately legal alternatives")
    version=policy.policy_id
    values={}
    for action in sorted(options):
        continuations=[
            frozen_continuation(env,state,action,policy,seed+104729*i)
            for i in range(n_continuations)
        ]
        if policy.policy_id!=version:
            raise RuntimeError("Policy version changed while measuring Q_pi")
        rewards=[x["reward"] for x in continuations]
        if compute_oracle:
            from .cce import native_future_value
            qstar=native_future_value(env,state,action)
        else:
            qstar=None
        values[action]={
            "mean_policy_reward":statistics.mean(rewards),
            "std_policy_reward":statistics.pstdev(rewards),
            "safe_successes":sum(x["safe_success"] for x in continuations),
            "violations":sum(x["ci_violation"] for x in continuations),
            "safe_failures":sum(x["safe_failure"] for x in continuations),
            "mean_terminal_steps":statistics.mean(x["final_steps"] for x in continuations),
            "rollouts":continuations,
            "qstar":qstar,
        }
    qpi=[v["mean_policy_reward"] for v in values.values()]
    return {
        "scenario":asdict(fork.scenario),
        "scenario_key":fork.scenario.key,
        "prefix":list(fork.prefix),
        "prefix_state_hash":fork.state_hash,
        "phase":fork.phase,"role":fork.role,
        "policy_id":version,"n_continuations_per_action":n_continuations,
        "seed":seed,"actions":values,
        "policy_return_range":max(qpi)-min(qpi),
        "qstar_is_ideal_not_expected_return":True,
        "no_oracle_in_model_observations":True,
        "all_candidates_immediately_valid":True,
    }

def sign_with_margin(a,b,margin:float)->int:
    if margin<0:raise ValueError("margin negative")
    return 1 if a-b>margin else -1 if b-a>margin else 0

def compare_oracle_policy(result,*,qstar_margin=.005,qpi_margin=.25)->dict:
    """Pairwise Qstar/Qpi agreement. Ties are not scored as agreement."""
    v=result["actions"]
    if len(v)!=2:raise ValueError("Expected exactly two candidate actions")
    x,y=sorted(v)
    a,b=v[x],v[y]
    if a["qstar"] is None or b["qstar"] is None:
        raise ValueError("No Qstar teacher measured")
    teacher=sign_with_margin(a["qstar"]["q"],b["qstar"]["q"],qstar_margin)
    empirical=sign_with_margin(a["mean_policy_reward"],b["mean_policy_reward"],qpi_margin)
    return {
        "teacher_preference":teacher,"policy_preference":empirical,
        "teacher_gap":a["qstar"]["q"]-b["qstar"]["q"],
        "policy_gap":a["mean_policy_reward"]-b["mean_policy_reward"],
        "ranking":("agree" if teacher and teacher==empirical else
                   "conflict" if teacher and empirical and teacher!=empirical else
                   "policy_tie" if teacher and not empirical else "teacher_tie"),
        "safe_success_counts":{k:v["safe_successes"] for k,v in v.items()},
        "violation_counts":{k:v["violations"] for k,v in v.items()},
    }
