"""Strict bridge from a generated canonical Action to finite dynamic tool calls."""
from copy import deepcopy
from .contracts import Action


def execute_tool_action(env,state,action):
    if not isinstance(action,Action):
        return deepcopy(state),{'verdict':'UNKNOWN','reason':'typed Action required'}
    if action.actor!=env.actor(state) or action.recipient or action.fact_ids or action.body or action.answer:
        return deepcopy(state),{'verdict':'UNKNOWN','reason':'actor or payload violates finite tool contract'}
    return env.transition(state,action.kind)
