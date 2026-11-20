"""Exact native finite-horizon Q* for V3.1.

Enumerates every future legal action, including read order, disclosure,
private verifier, aborted task and final business commit. Returns actual
native actions plus hashes and terminal business/delivery evidence.

This is deliberately NOT the frozen LLM policy Qpi.
"""
from __future__ import annotations
from dataclasses import asdict
from collections import Counter
from hashlib import sha256
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.diverse_ci_branch_env import digest,FINAL
from kci.diverse_ci_long_horizon_v31 import V31Env,V31Case

def reward(st,cost):
    return round(1.0-cost*st.steps,7) if st.success and not st.violation else -1.0 if st.violation else 0.0

def before_preflight(env):
    state=env.initial()
    state,ans=env.transition(state,"INSPECT_BUSINESS")
    assert ans["verdict"]=="VALID"
    route=env._route
    assert route not in ("summary","report_unavailable")
    state,ans=env.transition(state,"CHOOSE:"+route)
    assert ans["verdict"]=="VALID"
    assert state.stage=="preflight"
    assert {"OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"}<=set(env.options(state))
    return state,env.observation(state)

def native_best(env,start,chosen):
    """Exact native Q* after a given legal candidate action from one state."""
    assert chosen in env.options(start)
    memo={}
    counts=Counter()
    def dp(st):
        key=digest(asdict(st))
        if key in memo:return memo[key]
        counts["states"]+=1
        if st.done:
            v=(reward(st,env.spec.cost_per_step),(),st)
            memo[key]=v
            counts["end_states"]+=1
            return v
        best=None
        for a in env.options(st):
            nxt,report=env.transition(st,a)
            if report["verdict"] not in ("VALID","VIOLATION"):
                counts["unknown_or_invalid"]+=1
                continue
            counts["actual_transitions"]+=1
            assert nxt.steps>st.steps or nxt.done
            value,actions,terminal=dp(nxt)
            contender=(value,(a,)+actions,terminal)
            if best is None or value>best[0]+1e-8 or (
                abs(value-best[0])<1e-8 and
                (len(contender[1]),contender[1])<(len(best[1]),best[1])):
                best=contender
        if best is None:
            raise RuntimeError("No allowed terminal path from nonterminal state")
        memo[key]=best
        return best
    nxt,a=env.transition(start,chosen)
    assert a["verdict"] in ("VALID","VIOLATION")
    value,actions,end=dp(nxt)
    steps=[]
    curr=start
    for action in (chosen,)+actions:
        oldhash=digest(asdict(curr))
        new,audit=env.transition(curr,action)
        assert audit["verdict"] in ("VALID","VIOLATION")
        steps.append({
            "action":action,"role":env.observation(curr)["role"] if not curr.done else None,
            "state_before_sha256":oldhash,
            "state_after_sha256":digest(asdict(new)),
            "native_steps_after":new.steps,"stage_before":curr.stage,
            "stage_after":new.stage,"actual_verdict":audit["verdict"],
            "received_R":len(new.delivered["R"]),
            "received_M":len(new.delivered["M"]),
        })
        curr=new
    assert digest(asdict(curr))==digest(asdict(end))
    assert reward(curr,env.spec.cost_per_step)==value
    assert curr.done
    terminal={
        "success":curr.success,"violation":curr.violation,
        "safe_incomplete":not curr.success and not curr.violation,
        "steps":curr.steps,"reward":value,"business_final_hash":digest(curr.business),
        "business_result":curr.generated_result,
        "recipient_R_evidence":list(curr.delivered["R"]),
        "recipient_M_evidence":list(curr.delivered["M"]),
        "actual_received_message_log":curr.ledger,
        "actual_tool_calls":curr.tool_calls,
    }
    return {
        "first_action":chosen,"native_exact_Qstar":value,
        "future_continuations":len(memo),
        "states_explored":counts["states"],
        "actual_transitions_evaluated":counts["actual_transitions"],
        "action_trace":steps,
        "terminal":terminal,
        "provenance":"EXACT_DETERMINISTIC_NATIVE_MDP_V31_NOT_LLM_QPI"
    }

def pair_values(env):
    state,obs=before_preflight(env)
    open_result=native_best(env,state,"OPEN_PRIVATE_CHANNEL")
    skip_result=native_best(env,state,"SKIP_PRIVATE_CHANNEL")
    return {
        "case_id":env.spec.case_id,"root":env.task.id,
        "graph":env.task.graph,"business":env.task.business,
        "condition":env.condition,"history_kind":env.history_kind,
        "authorized":env.authorized,
        "state_hash":digest(asdict(state)),
        "model_observation":obs,
        "model_observation_sha256":digest(obs),
        "open":open_result,"skip":skip_result,
        "Qgap_open_minus_skip":round(open_result["native_exact_Qstar"]-
                                      skip_result["native_exact_Qstar"],7),
    }
