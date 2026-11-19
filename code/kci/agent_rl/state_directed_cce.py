"""State-directed PCCE credit: frozen Qpi of SAME VERIFIED restart state.

A sampled policy chooses the restart action on-policy (it is NOT forced to take
the teacher-preferred action). Both valid candidate continuations are sampled
under the SAME frozen policy before ANY optimizer update. Only the model's
first action token receives centered local credit; subsequent role actions
retain terminal GRPO credit. Unknown, unsafe and ambiguous evidence fail closed.
"""
from __future__ import annotations
from collections import Counter
import math

from .core import Episode,digest
from .state_restarts import RestartSpec,restore
from .policy_continuation import immediately_legal,frozen_policy_counterfactual
from .cce import EARLY


def robust_pair_signal(result:dict,*,n:int,minimum_gap:float=.5,
                       minimum_consensus:float=.75)->dict:
    if not 0<minimum_gap<=2 or not .5<minimum_consensus<=1:
        raise ValueError("Invalid frozen-policy CCE confidence gate")
    options=result["actions"]
    if len(options)!=2:
        raise ValueError("Robust CCE needs two immediately valid actions")
    a,b=sorted(options)
    left,right=options[a]["rollouts"],options[b]["rollouts"]
    if len(left)!=n or len(right)!=n:
        raise ValueError("Incomplete frozen Qpi continuation sample")
    gaps=[x["reward"]-y["reward"] for x,y in zip(left,right)]
    pos=sum(d>.1 for d in gaps)
    neg=sum(d<-.1 for d in gaps)
    consensus=max(pos,neg)/n
    mean=options[a]["mean_policy_reward"]-options[b]["mean_policy_reward"]
    gap_ok=abs(mean)>=minimum_gap
    consensus_ok=consensus>=minimum_consensus
    return {"supported":gap_ok and consensus_ok,
            "mean_reward_gap":mean,"paired_sign_consensus":consensus,
            "pos_paired":pos,"neg_paired":neg,
            "gap_pass":gap_ok,"consensus_pass":consensus_ok,
            "min_gap":minimum_gap,"min_consensus":minimum_consensus}


class StateDirectedPCCE:
    def __init__(self, *,source_root=None,n_continuations=4,
                 minimum_gap=.5,minimum_consensus=.75,seed=281119):
        if not 2<=n_continuations<=16:raise ValueError("Invalid K")
        if not 0<minimum_gap<=2 or not .5<minimum_consensus<=1:
            raise ValueError("Invalid signal gate")
        self.source_root=source_root
        self.k=n_continuations
        self.minimum_gap=minimum_gap
        self.minimum_consensus=minimum_consensus
        self.seed=seed

    def make_credits(self,spec:RestartSpec,group:list[Episode],policy,*,group_index:int):
        if not group or group_index<0:
            raise ValueError("Missing restart group")
        if any(e.scenario_key!=spec.scenario.key or
               e.initial_state_hash!=spec.state_hash or
               e.policy_id!=policy.policy_id or not e.decisions or
               e.decisions[0].state_hash!=spec.state_hash for e in group):
            raise RuntimeError("Group does not consist of on-policy draws from this restart")
        env,state=restore(spec,self.source_root)
        status=(spec.role,spec.phase)
        counters=Counter()
        counters["rollouts"]=len(group)
        if status not in EARLY:
            return ([{} for _ in group],[],{"skipped_not_early":len(group)})
        if len(immediately_legal(env,state))!=2:
            return ([{} for _ in group],[],{"skipped_less_than_two_legal":len(group)})
        version=policy.policy_id
        result=frozen_policy_counterfactual(
            spec.to_fork(),policy,
            n_continuations=self.k,
            seed=self.seed+1000003*group_index,
            source_root=self.source_root,
            compute_oracle=False,
            require_early=True)
        if result["policy_id"]!=version or policy.policy_id!=version:
            raise RuntimeError("Frozen Qpi policy changed during branch sampling")
        gate=robust_pair_signal(
            result,n=self.k,minimum_gap=self.minimum_gap,
            minimum_consensus=self.minimum_consensus)
        result["signal_gate"]=gate
        counters["queried_prefixes"]=1
        counters["policy_branch_rollouts"]=2*self.k
        counters["supported_prefixes"]=int(gate["supported"])
        counters["rejected_small_gap"]=int(not gate["gap_pass"])
        counters["rejected_low_consensus"]=int(not gate["consensus_pass"])
        actions=result["actions"]
        values=[v["mean_policy_reward"] for v in actions.values()]
        center=sum(values)/len(values)
        credits=[]
        proofs=[]
        for e in group:
            d=e.decisions[0]
            if not gate["supported"] or d.chosen_action not in actions:
                credits.append({})
                continue
            local=max(-1.,min(1.,actions[d.chosen_action]["mean_policy_reward"]-center))
            if abs(local)<1e-8:
                credits.append({})
                continue
            credits.append({0:local})
            counters["credited_decisions"]+=1
            counters["positive_credits"]+=int(local>0)
            counters["negative_credits"]+=int(local<0)
            proofs.append({
                "group_index":group_index,
                "sample_index":e.sample_index,
                "decision_index":0,
                "scenario":spec.scenario.key,
                "restart_prefix":list(spec.prefix),
                "restart_state_hash":spec.state_hash,
                "chosen_action":d.chosen_action,
                "role":spec.role,"phase":spec.phase,
                "local_credit":local,
                "policy_id":version,
                "teacher_kind":"frozen_policy_Qpi_from_native_restart",
                "native_Qstar_used":False,
                "Qpi_evidence":result,
                "source_contract":"auditable_native_delivered_receipts_only",
            })
        return credits,proofs,dict(counters)
