"""KCI CCE upper-bound teacher for ACTING-agent action credit.

This diagnostic CCE uses the **exact native synthetic continuation oracle Q***.
It is not an estimate of current-policy return, and must NEVER appear in
policy prompts or at autonomous inference. The original terminal reward,
native executable transitions, and user task contract are unchanged.

One same-prefix counterfactual comparison per distinct eligible state:
- both candidate actions must be currently legal;
- evaluate native exact safe-completion feasibility from each successor;
- shape ONLY the chosen acting-role token's training advantage.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import math

from .core import Episode, Scenario, digest
from kci.dynamic_tasks import canonical

EARLY = {
    ("Planner","choose"),
    ("Broker","choose"),
    ("Broker","collect"),
    ("HR","hr"),
}
EXCLUDED = {"ABORT","WITHHOLD"}

def native_future_value(env,state,action):
    next_state, verdict = env.transition(state,action)
    if verdict["verdict"]=="UNKNOWN":
        raise RuntimeError("Unexpected UNKNOWN counterfactual transition")
    if verdict["verdict"]=="VIOLATION":
        return {"legal":False,"reachable":False,"q":-1.0,
                "optimal_terminal_turn":None}
    proof = env.exact(next_state)
    if proof["status"]=="PROVEN_UNREACHABLE":
        return {"legal":True,"reachable":False,"q":0.,
                "optimal_terminal_turn":None}
    if proof["status"]!="EXACT":
        raise RuntimeError("Native exact verifier did not return a definitive result")
    cost=int(proof["min_total_cost"])
    if not 1<=cost<=env.horizon:
        raise RuntimeError("Native continuation cost outside fixed horizon")
    return {"legal":True,"reachable":True,
            "q":1.0-0.01*cost,
            "optimal_terminal_turn":cost}


class NativeCCEOracle:
    def __init__(self, *, source_root=None, min_q_gap=0.005):
        if not 0<min_q_gap<=1:raise ValueError("Invalid Q gap threshold")
        self.source_root=source_root
        self.min_q_gap=min_q_gap
        self._cache={}
        self.calls=0

    def make_credits(self, scenario: Scenario, group: list[Episode]):
        if not group or len({e.scenario_key for e in group})!=1:
            raise ValueError("Group needs one consistent scenario")
        if any(e.scenario_key!=scenario.key for e in group):
            raise ValueError("Scenario mismatch")
        credits=[]
        proofs=[]
        counters=Counter()
        for ep in group:
            env=(scenario.build() if self.source_root is None
                 else scenario.build(self.source_root))
            state=env.decision_start()
            if digest(asdict(state))!=ep.initial_state_hash:
                raise RuntimeError("CCE replay starting state mismatch")
            ecredit={}
            for t,decision in enumerate(ep.decisions):
                if digest(asdict(state))!=decision.state_hash:
                    raise RuntimeError("CCE replay diverged from actual sampled policy prefix")
                actions=tuple(env.options(state))
                if decision.chosen_action not in actions:
                    raise RuntimeError("CCE action not executable")
                obs=env.observation(state)
                if decision.observation_hash!=digest(obs):
                    raise RuntimeError("CCE policy observation mismatch")
                if (env.actor(state),obs["phase"]) in EARLY and len(actions)>=2:
                    key=(scenario.key,decision.state_hash)
                    if key not in self._cache:
                        self.calls+=1
                        self._cache[key]={
                            a:native_future_value(env,state,a)
                            for a in actions
                        }
                    values=self._cache[key]
                    admissible={a:v["q"] for a,v in values.items()
                                if v["legal"] and a not in EXCLUDED}
                    chosen=decision.chosen_action
                    if len(admissible)>=2:
                        magnitude=max(admissible.values())-min(admissible.values())
                        if magnitude>=self.min_q_gap:
                            if chosen in admissible:
                                mean=sum(admissible.values())/len(admissible)
                                value=max(-1.,min(1.,2*(admissible[chosen]-mean)/magnitude))
                                if abs(value)<=1e-8:
                                    counters["zero_centered_credits_excluded"]+=1
                                else:
                                    ecredit[t]=value
                                    counters["eligible_decisions"]+=1
                                    counters["nonzero_credits"]+=1
                                    counters["positive_credits"]+=value>1e-8
                                    counters["negative_credits"]+=value< -1e-8
                                    counters["policy_action_was_legal"]+=1
                                    proofs.append({"scenario":scenario.key,
                                        "sample_index":ep.sample_index,
                                        "decision_index":t,
                                        "role":decision.actor,"phase":obs["phase"],
                                        "state_hash":decision.state_hash,
                                        "observation_hash":decision.observation_hash,
                                        "chosen":chosen,"score":value,
                                        "teacher":"native_synthetic_exact_Qstar",
                                        "alternatives":values,
                                        "exclude_oracle_from_inference":True})
                            else:
                                counters["chosen_immediate_illegal"]+=1
                        else:
                            counters["insufficient_future_gap"]+=1
                    else:
                        counters["insufficient_legal_alternatives"]+=1
                state,verdict=env.transition(state,decision.chosen_action)
                if verdict["verdict"]!=decision.verdict:
                    raise RuntimeError("CCE changed actual native verdict")
            if not state.done or state.success!=ep.success or state.violation!=ep.violation:
                raise RuntimeError("CCE replay final state disagrees with real trajectory")
            credits.append(ecredit)
        counters["episodes"]=len(group)
        counters["distinct_cached_teacher_states"]=len(self._cache)
        counters["exact_query_state_calls"]=self.calls
        return credits,proofs,dict(counters)


def score_credit(terminal_advantage: float, local_credit: float | None, weight: float):
    if not 0<=weight<=1:raise ValueError("CCE blend must be 0..1")
    if local_credit is None:
        return float(terminal_advantage)
    if not all(math.isfinite(x) for x in (terminal_advantage,local_credit)):
        raise ValueError("Non-finite CCE credit")
    if not -1.000001<=local_credit<=1.000001:
        raise ValueError("Native CCE advantage must be bounded")
    return (1-weight)*terminal_advantage+weight*local_credit
