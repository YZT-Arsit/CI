"""Policy-Calibrated Counterfactual Credit (PCCE), bounded pilot implementation.

Unlike the rejected native Qstar teacher, this samples COMPLETE continuations
from the SAME current LM before its optimizer step. It estimates Q_pi(s,a)
for both immediately valid actions. Local credit is the RAW return difference
from the action-pair average, NOT normalized by a 0.01 exact-cost spread.

The existing terminal GRPO advantage must be PRESERVED additively; do not
use the older convex mixture on this return-based credit.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import math
from .core import Episode,Scenario,digest
from .cce import EARLY
from .policy_continuation import (
    FrozenFork,replay_prefix,immediately_legal,frozen_policy_counterfactual
)

class PolicyCCEEstimator:
    def __init__(self, *, source_root=None,n_continuations=4,
                 min_mean_gap=.5,min_consensus=.75,max_prefixes_per_group=4,
                 seed=260119):
        if not 2<=n_continuations<=16:raise ValueError("n_continuations must be 2..16")
        if not 0<min_mean_gap<=2:raise ValueError("Expected return gap outside (0,2]")
        if not .5<min_consensus<=1:raise ValueError("Consensus outside (.5,1]")
        if not 1<=max_prefixes_per_group<=16:raise ValueError("Too many queried forks")
        self.source_root=source_root
        self.n=n_continuations
        self.min_gap=min_mean_gap
        self.min_consensus=min_consensus
        self.limit=max_prefixes_per_group
        self.seed=seed

    def make_credits(self,scenario:Scenario,group:list[Episode],policy,*,group_index:int):
        if not group or any(e.scenario_key!=scenario.key for e in group):
            raise ValueError("Counterfactual group world mismatch")
        policy_id=policy.policy_id
        if any(e.policy_id!=policy_id for e in group):
            raise ValueError("Stale/mixed on-policy samples")
        if group_index<0:raise ValueError("Invalid sampling group index")
        cache={}
        records=[]
        credits=[]
        stats=Counter()
        for episode in group:
            env,state=replay_prefix(scenario,(),self.source_root)
            if digest(asdict(state))!=episode.initial_state_hash:
                raise RuntimeError("Actual prefix initial state mismatch")
            actions_taken=[]
            ecredit={}
            for step,d in enumerate(episode.decisions):
                if digest(asdict(state))!=d.state_hash:
                    raise RuntimeError("Replay state hash diverged")
                if env.actor(state)!=d.actor:
                    raise RuntimeError("Replay acting role diverged")
                if digest(env.observation(state))!=d.observation_hash:
                    raise RuntimeError("Replay observation diverged")
                current=(d.actor,env.observation(state)["phase"])
                valid=tuple(env.options(state))
                if d.chosen_action not in valid:
                    raise RuntimeError("Recorded action not executable")
                if current in EARLY and len(valid)>=2:
                    key=(scenario.key,d.state_hash)
                    if key not in cache and len(cache)<self.limit:
                        legal=immediately_legal(env,state)
                        if len(legal)==2:
                            fork=FrozenFork(scenario,tuple(actions_taken),
                                            current[1],current[0],d.state_hash)
                            fork_seed=self.seed+1000003*group_index + (
                                int(sha256(key[1].encode()).hexdigest()[:10],16)%1000000)
                            result=frozen_policy_counterfactual(
                                fork,policy,
                                n_continuations=self.n,
                                seed=fork_seed,
                                source_root=self.source_root,
                                compute_oracle=False,
                                require_early=True)
                            if result["policy_id"]!=policy_id:
                                raise RuntimeError("Policy mutated during CCE continuation")
                            alt=result["actions"]
                            a,b=sorted(alt)
                            qa,qb=alt[a]["mean_policy_reward"],alt[b]["mean_policy_reward"]
                            paired=[x["reward"]-y["reward"] for x,y in
                                    zip(alt[a]["rollouts"],alt[b]["rollouts"])]
                            positive=sum(x>.1 for x in paired)
                            negative=sum(x<-.1 for x in paired)
                            agree=max(positive,negative)/self.n
                            margin=qa-qb
                            gap_ok=abs(margin)>=self.min_gap
                            consensus_ok=agree>=self.min_consensus
                            supported=gap_ok and consensus_ok
                            stats["rejected_small_return_gap"]+=int(not gap_ok)
                            stats["rejected_low_direction_consensus"]+=int(not consensus_ok)
                            stats["rejected_both"]+=int(not gap_ok and not consensus_ok)
                            result["signal_gate"]={
                                "supported":supported,
                                "mean_return_gap":margin,
                                "pairwise_consensus":agree,
                                "positive_paired_samples":positive,
                                "negative_paired_samples":negative,
                                "minimum_mean_return_gap":self.min_gap,
                                "minimum_consensus":self.min_consensus,
                                "terminal_reward_not_replaced":True
                            }
                            cache[key]=result
                            stats["queried_prefixes"]+=1
                            stats["counterfactual_rollouts"]+=2*self.n
                            stats["supported_prefixes"]+=int(supported)
                        else:
                            stats["insufficient_legal_alternatives"]+=1
                    if key in cache:
                        result=cache[key]
                        if result["signal_gate"]["supported"] and d.chosen_action in result["actions"]:
                            q=result["actions"]
                            values=[v["mean_policy_reward"] for v in q.values()]
                            avg=sum(values)/len(values)
                            raw=q[d.chosen_action]["mean_policy_reward"]-avg
                            local=max(-1.,min(1.,raw))
                            if abs(local)>1e-8:
                                ecredit[step]=local
                                stats["credit_decisions"]+=1
                                stats["positive_credits"]+=int(local>0)
                                stats["negative_credits"]+=int(local<0)
                                records.append({
                                    "group_index":group_index,
                                    "scenario":scenario.key,
                                    "sample_index":episode.sample_index,
                                    "decision_index":step,
                                    "state_hash":d.state_hash,
                                    "prefix":list(actions_taken),
                                    "role":d.actor,"phase":current[1],
                                    "chosen_action":d.chosen_action,
                                    "local_credit":local,
                                    "policy_id":policy_id,
                                    "teacher_kind":"frozen_same_step_policy_Qpi",
                                    "no_native_Qstar_used":True,
                                    "action_returns":result,
                                })
                nxt,audit=env.transition(state,d.chosen_action)
                if audit["verdict"]!=d.verdict:
                    raise RuntimeError("Recorded native verdict changed")
                actions_taken.append(d.chosen_action)
                state=nxt
            if not state.done or state.success!=episode.success or state.violation!=episode.violation:
                raise RuntimeError("Native rollout did not replay")
            credits.append(ecredit)
        if policy.policy_id!=policy_id:
            raise RuntimeError("Policy version changed while estimating Qpi")
        stats["episodes"]=len(group)
        stats["max_unique_prefixes"]=self.limit
        stats["n_continuations_per_action"]=self.n
        return credits,records,dict(stats)
