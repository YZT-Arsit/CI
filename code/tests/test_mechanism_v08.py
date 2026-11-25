"""Bellman safety, tied-success cost accounting and same-prompt intervention checks."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kci.mechanism_v08 import Planner, compare_world, direction
from kci.paper_worlds import PhaseWorld, phase


def test_currently_legal_resource_setup_changes_future_reachable_completion():
    w=PhaseWorld(next(s for s in phase.make_workflows() if s.id=="P202"))
    result=compare_world(w)
    assert result["probes"]["missing"]["verdict"]=="VALID"
    assert result["probes"]["present"]["inferred"] and not result["probes"]["present"]["direct"]
    histories=result["history"]
    assert histories["present"]["options"]==histories["missing"]["options"]
    assert histories["present"]["actor"]==histories["missing"]["actor"]
    assert histories["present"]["starting_cost"]==histories["missing"]["starting_cost"]
    assert histories["present"]["q"]["optimal"]!=histories["missing"]["q"]["optimal"]


def test_horizon_expiry_and_abort_have_zero_value_and_unsafe_action_ineligible():
    w=PhaseWorld(next(s for s in phase.make_workflows() if s.id=="P202"))
    s,_=phase.native_source_prefix(w.env)
    plan=Planner(w)
    assert plan.value(s,0,"optimal")== (0,0,0)
    assert plan.action(s,"ABORT",10,"optimal")== (0,0,0)
    assert plan.action(s,"SEND_RAW_TARGET",10,"optimal") is None
    assert plan.action(s,"SEND_RAW_TARGET",10,"reference")[0]==1


def test_scalar_completion_ties_can_still_have_a_conditional_cost_gap():
    q={"q":{"optimal":[["0","1","9"],["0","1","7"]]}}
    assert direction(q,"optimal")==-1
    q["q"]["optimal"][0][1]="3/4"
    q["q"]["optimal"][0][2]="1"
    assert direction(q,"optimal")==-1

