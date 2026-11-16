"""CCE: finite reference continuation and conservative preference certification.

Exhaustive means exhaustive ONLY relative to the declared candidate action space,
known transition model, and bounded horizon. A model generator is never exhaustive.
"""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable
import json
from .contracts import (Action, CandidateBatch, CandidateGenerator, CCEReport, Observation,
                        Preference, RefResult, RefStatus, State, StepAudit,Trace,TraceStep,Verdict)

@dataclass(frozen=True)
class SearchBudget:
    max_nodes: int=2000
    max_actions: int=12

class ReferenceSearch:
    def __init__(self,env,generator:CandidateGenerator,budget:SearchBudget=SearchBudget()):
        self.env=env;self.generator=generator;self.budget=budget
        self._cache={}

    def search(self,start:State)->RefResult:
        # Search always starts AFTER the candidate action; includes that action's
        # cost in start.cost. Never start reference search from a leaked state.
        cache_key=(self.env.version,self.generator.version,self.budget,
                   start.signature())
        use_cache=bool(getattr(self.generator,'deterministic',False))
        if use_cache and cache_key in self._cache:
            return deepcopy(self._cache[cache_key])
        start=deepcopy(start)
        if start.any_violation:
            return RefResult(RefStatus.UNKNOWN,None,None,0,'state already has a CI violation',
                             self.env.version,self.generator.version,start.max_turns,
                             (self.budget.max_nodes,self.budget.max_actions))
        queue=deque([(start,tuple())]); expanded=0;complete=True
        best_trace=None;best_cost=None
        while queue:
            if expanded>=self.budget.max_nodes:
                complete=False;break
            st,steps=queue.popleft();expanded+=1
            if st.done:
                if st.success and not st.any_violation:
                    if best_cost is None or st.cost < best_cost:
                        best_cost=st.cost;best_trace=Trace(steps,start_state=st if not steps else None)
                continue
            if st.turn>=st.max_turns:continue
            batch=self.generator.generate(self.env.observation(st,st.actor),limit=self.budget.max_actions)
            if not batch.exhaustive:
                complete=False
            for act in batch.actions:
                next_state,audit=self.env.transition(st,act)
                if audit.verdict in (Verdict.UNKNOWN,Verdict.INVALID_ACTION):
                    # Unknown semantic meaning prevents certification of optimality.
                    if audit.verdict==Verdict.UNKNOWN:complete=False
                    continue
                if audit.verdict==Verdict.VIOLATION:continue
                step=TraceStep(st,act,audit,next_state)
                queue.append((next_state,steps+(step,)))
        if best_trace is not None:
            status=RefStatus.EXACT if complete else RefStatus.WITNESS_ONLY
        else:
            status=RefStatus.PROVEN_UNREACHABLE if complete else RefStatus.UNKNOWN
        why=('bounded exhaustive search over declared finite actions' if complete
             else 'incomplete action coverage, node budget, or semantic uncertainty')
        result=RefResult(status,best_trace,best_cost,expanded,why,
                          self.env.version,self.generator.version,start.max_turns,
                          (self.budget.max_nodes,self.budget.max_actions))
        if use_cache:self._cache[cache_key]=deepcopy(result)
        return result

class SeededPolicy:
    """Stochastic frozen policy adapter. Same external seed can be reused
    across same-prefix counterfactual branches as common random numbers.
    """
    def __init__(self,sample:Callable[[Observation,int],Action],version:str):
        self._sample=sample;self.version=version
    def sample(self,obs:Observation,seed:int)->Action:
        return self._sample(obs,seed)

class DeterministicPolicy:
    """Adapter for a fixed snapshot of an actual model; for the demo, a callable."""
    def __init__(self,choose:Callable[[Observation],Action],version:str):
        self.choose=choose;self.version=version
    def sample(self,obs:Observation,seed:int)->Action:
        return self.choose(obs)

class ContinuationEvaluator:
    def __init__(self,env,reference:ReferenceSearch|None,actual_policy:DeterministicPolicy|None=None):
        if reference is not None and env.version!=reference.env.version:
            raise ValueError('CCE and reference search require the same environment version')
        self.env=env;self.reference=reference;self.actual_policy=actual_policy

    @staticmethod
    def state_id(state:State)->str:
        raw=repr(state.signature()).encode('utf-8')
        return sha256(raw).hexdigest()[:16]

    def rollout_actual(self,state:State,seed:int=0)->Trace:
        if self.actual_policy is None:raise ValueError('actual policy missing')
        steps=[];st=deepcopy(state)
        while not st.done and st.turn<st.max_turns:
            obs=self.env.observation(st,st.actor)
            action=self.actual_policy.sample(obs,seed + 10007*st.turn)
            next_state,audit=self.env.transition(st,action)
            steps.append(TraceStep(st,action,audit,next_state))
            if audit.verdict in (Verdict.UNKNOWN,Verdict.INVALID_ACTION,Verdict.VIOLATION):
                break
            st=next_state
        return Trace(tuple(steps),start_state=st if not steps else None)

    def evaluate(self,state:State,action:Action,actual_rollouts:int=0,seed_base:int=17)->CCEReport:
        if self.reference is None:raise ValueError('reference search missing')
        obs=self.env.observation(state,state.actor)
        next_state,audit=self.env.transition(state,action)
        ref=None;actual=()
        if audit.verdict==Verdict.VALID:
            ref=self.reference.search(next_state)
            if self.actual_policy is not None and actual_rollouts>0:
                # Across candidate branches, reuse seed_base and rollout index
                # to reduce randomness-induced false preference differences.
                actual=tuple(self.rollout_actual(next_state,seed_base+31*j) for j in range(actual_rollouts))
        return CCEReport(state_id=self.state_id(state),observation=obs,action=action,
                         immediate=audit,reference=ref,actual_traces=actual,
                         policy_version=self.actual_policy.version if self.actual_policy else 'none',
                         reference_version=self.reference.env.version,
                         generator_version=self.reference.generator.version,
                         search_budget=(self.reference.budget.max_nodes,self.reference.budget.max_actions),
                         horizon=state.max_turns)


def certify_preference(a:CCEReport,b:CCEReport,*,max_cost_gap:float=0.0)->Preference|None:
    """Strictly evidence-backed comparisons; no negative labels from search misses.

    Note: for safe-vs-unsafe, the safe alternative must have a VERIFIED SUCCESS
    witness. This prevents choosing empty refusal as a cheap shortcut.
    """
    if a.state_id!=b.state_id or a.observation.actor!=b.observation.actor:
        raise ValueError('Comparisons must share the exact pre-action state/actor')
    if a.observation!=b.observation or a.observation.as_prompt()!=b.observation.as_prompt():
        raise ValueError('Comparisons must share the exact authorized observation and prompt')
    if a.action.actor!=a.observation.actor or b.action.actor!=b.observation.actor:
        raise ValueError('Candidate actor must match the observation')
    contexts=[]
    for report in (a,b):
        context=(report.reference_version,report.generator_version,report.search_budget,report.horizon)
        if not report.reference_version or not report.generator_version or \
           report.search_budget is None or report.horizon is None:
            return None
        contexts.append(context)
        if report.reference is not None:
            ref=report.reference
            if context!=(ref.verifier_version,ref.generator_version,ref.search_budget,ref.horizon):
                raise ValueError('Reference result does not match its recorded comparison context')
    if contexts[0]!=contexts[1] or a.policy_version!=b.policy_version:
        raise ValueError('Comparisons require the same reference versions, budget, horizon and policy snapshot')
    if max_cost_gap<0:raise ValueError('max_cost_gap must be nonnegative')
    provenance={'reference_version':a.reference_version,'generator_version':a.generator_version,
                'search_budget':{'max_nodes':a.search_budget[0],'max_actions':a.search_budget[1]},
                'horizon':a.horizon,'policy_version':a.policy_version}
    if a.action.key()==b.action.key():return None
    allowed={Verdict.VALID,Verdict.VIOLATION}
    if a.immediate.verdict not in allowed or b.immediate.verdict not in allowed:return None
    for good,bad in ((a,b),(b,a)):
        if (good.immediate.verdict==Verdict.VALID and
            bad.immediate.verdict==Verdict.VIOLATION and
            good.reference and good.reference.best_trace is not None):
            return Preference(good.observation,good.action,bad.action,'ci_safe_success',good.state_id,
                {'certified':True,'evidence_type':'direct_or_incremental_violation_plus_verified_safe_witness',
                 'violations':list(bad.immediate.direct_evidence+bad.immediate.inference_evidence),
                 'reference_status':good.reference.status.value,
                 **provenance})
    if a.immediate.verdict==Verdict.VIOLATION or b.immediate.verdict==Verdict.VIOLATION:
        return None
    assert a.reference is not None and b.reference is not None
    ra,rb=a.reference,b.reference
    # Both values exact within the explicit finite model/horizon.
    if ra.status==RefStatus.EXACT and rb.status==RefStatus.EXACT:
        diff=(rb.best_success_cost or 0)-(ra.best_success_cost or 0)
        if abs(diff)<=max_cost_gap:return None
        good,bad=(a,b) if diff>0 else (b,a)
        return Preference(good.observation,good.action,bad.action,'future_verified',good.state_id,
            {'certified':True,'evidence_type':'exhaustive_bounded_reference_cost_comparison',
             'chosen_cost':good.reference.best_success_cost,
             'rejected_cost':bad.reference.best_success_cost,
             **provenance})
    # Reachability dominance: a verified feasible path vs proven infeasibility.
    for good,bad in ((a,b),(b,a)):
        if (good.reference.best_trace is not None and
            bad.reference.status==RefStatus.PROVEN_UNREACHABLE):
            return Preference(good.observation,good.action,bad.action,'future_reachable',good.state_id,
               {'certified':True,'evidence_type':'success_witness_vs_exhaustively_proven_unreachable',
                **provenance})
    return None
