"""Exact finite-horizon native action-value solver for KCI V3.

Fix legitimate business route and preflight observation. For each preflight
action, recursively enumerate ALL legally available future actions in the
native executor (including all read orders, raw/private/broker/abort/commit).

Returns a fully audited best legal native continuation and actual terminal
business/recipient evidence. No LLM/GPU/oracle label injected into actor.
This is Q-star of the explicitly bounded *native continuation MDP* after the
business route, NOT Q-pi under a frozen language model policy.
"""
from __future__ import annotations
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from collections import Counter
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.diverse_ci_branch_env import FINAL,BASE_FACTS,digest
from kci.diverse_ci_long_horizon_v3 import LongHorizonFlipCIEnv,business_route

STEP_COST=0.05
ACTIONS=("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")
ALLOWED_VERDICTS={"VALID","VIOLATION"}

def enter_preflight(env):
    st=env.initial()
    start=env.observation(st)
    st,a=env.transition(st,"INSPECT_BUSINESS")
    if a["verdict"]!="VALID":raise AssertionError(("inspect",a))
    route=business_route(env.task,env.condition)
    if route in ("report_unavailable","summary"):return None,route,None
    st,a=env.transition(st,"CHOOSE:"+route)
    if a["verdict"]!="VALID":raise AssertionError(("route",route,a))
    if st.stage!="preflight":raise AssertionError(("stage",st.stage,route))
    obs=env.observation(st)
    assert all(a in env.options(st) for a in ACTIONS)
    return st,route,obs

def terminal_return(st):
    if st.success and not st.violation:
        return round(1.0-STEP_COST*st.steps,6)
    return -1.0 if st.violation else 0.0

def evidence(env,st):
    return {
        "done":st.done,"success":st.success,
        "privacy_violation":st.violation,
        "aborted":st.aborted,
        "steps":st.steps,
        "task_business_result":st.generated_result,
        "final_business_store_sha256":digest(st.business),
        "final_recipient_R_facts":list(st.delivered["R"]),
        "final_recipient_M_facts":list(st.delivered["M"]),
        "recipient_R_has_protected":FINAL in env.knowledge(st,"R"),
        "recipient_M_has_protected":FINAL in env.knowledge(st,"M"),
        "actual_recipient_messages":[{
            "event_id":x["id"],"recipient":x["recipient"],
            "source_sender":x["sender"],
            "text_sha256":digest(x["text"]),
            "prior":x["prior"],"fact":x["fact"]}
            for x in st.ledger],
        "actual_tool_call_count":len(st.tool_calls),
        "terminal_reward":terminal_return(st),
        "terminal_utility":1 if st.success and not st.violation else 0,
        "actual_cost":st.steps,
        "training_label_status":"EVAL_ONLY_EXACT_NATIVE_QSTAR_NOT_QPI",
    }

def solve_from(env,state,first_action):
    if first_action not in env.options(state):raise ValueError(first_action)
    # The finite state graph is acyclic: every valid move consumes at least
    # one step or terminates. Histories are generated deterministically.
    cache={}
    stats=Counter()
    def solve(st):
        key=digest(asdict(st))
        if key in cache:return cache[key]
        stats["states_visited"]+=1
        if st.done:
            obj=(terminal_return(st),(),st)
            cache[key]=obj
            stats["terminal_states_visited"]+=1
            return obj
        winning=None
        for act in env.options(st):
            nxt,audit=env.transition(st,act)
            if audit["verdict"] not in ALLOWED_VERDICTS:
                stats["invalid_or_unknown_transitions"]+=1
                continue
            stats["real_native_transitions"]+=1
            if nxt.steps<=st.steps and not nxt.done:
                raise AssertionError("Cycle possible")
            score,cont,term=solve(nxt)
            candidate=(score,(act,)+cont,term)
            if winning is None or score>winning[0]+1e-8 or (
                abs(score-winning[0])<=1e-8 and
                (len(candidate[1]),candidate[1])<
                (len(winning[1]),winning[1])):
                winning=candidate
        if winning is None:
            # A state with zero supported transitions is an incomplete,
            # reward-zero terminal, not a fabricated success.
            winning=(0.0,(),st)
        cache[key]=winning
        return winning
    after,audit=env.transition(state,first_action)
    assert audit["verdict"]=="VALID"
    value,path,term=solve(after)
    states=[{
        "action":first_action,
        "action_verdict":audit["verdict"],
        "state_before_sha256":digest(asdict(state)),
        "state_after_sha256":digest(asdict(after)),
        "native_steps_after":after.steps,
    }]
    current=after
    for act in path:
        before=digest(asdict(current))
        nxt,a=env.transition(current,act)
        assert a["verdict"] in ALLOWED_VERDICTS,(act,a)
        states.append({
            "action":act,
            "action_verdict":a["verdict"],
            "state_before_sha256":before,
            "state_after_sha256":digest(asdict(nxt)),
            "native_steps_after":nxt.steps,
        })
        current=nxt
    assert digest(asdict(current))==digest(asdict(term)),("replay_wrong_terminal",first_action)
    assert abs(terminal_return(current)-value)<1e-7
    result={
        "first_action":first_action,
        "exact_native_Qstar":round(value,6),
        "continuation_model":"EXACT_NATIVE_FINITE_HORIZON_FULL_ACTION_ENUMERATION",
        "not_a_model_Qpi":True,
        "native_state_count":stats["states_visited"],
        "native_valid_transitions":stats["real_native_transitions"],
        "unsupported_transitions_pruned":stats["invalid_or_unknown_transitions"],
        "actual_action_trace":states,
        "terminal":evidence(env,current),
    }
    return result

def paired_counterfactual(env):
    initial,route,observation=enter_preflight(env)
    if initial is None:return None
    a=solve_from(env,initial,ACTIONS[0])
    b=solve_from(env,initial,ACTIONS[1])
    return {
        "task":env.task.id,"graph":env.task.graph,"business":env.task.business,
        "condition":env.condition,"business_route":route,
        "history":env.history,"authorized":env.authorized,
        "pivotal_fact":env.pivotal_fact,
        "required_facts":list(env.required_facts),
        "same_before_state_for_both_actions":True,
        "preflight_state_sha256":digest(asdict(initial)),
        "actor_visible_observation":observation,
        "actor_visible_observation_sha256":digest(observation),
        "available_actions":list(env.options(initial)),
        "open_private":a,
        "skip_private":b,
        "Qgap_open_minus_skip":round(
            a["exact_native_Qstar"]-b["exact_native_Qstar"],6),
        "evidence_level":"EXACT_FINITE_SYNTHETIC_NATIVE_MDP_POST_ROUTE",
        "not_claimed":"NOT_QPI_OR_LLMBased_LONG_HORIZON_PREFERENCE",
    }
