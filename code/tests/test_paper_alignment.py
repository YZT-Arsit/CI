"""Regression checks for the native CCE/DPR contract, without a GPU checkpoint."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kci.paper_alignment import Alignment
from kci.paper_worlds import FINAL, PhaseWorld, key, phase


def world(**kwargs):
    spec = next(s for s in phase.make_workflows() if s.id == "P202")
    return PhaseWorld(spec, **kwargs)


@pytest.mark.parametrize("variant,authorized,expected", [
    ("risky", False, True), ("risky", True, False), ("safe", False, False),
    ("hold", False, False),
])
def test_same_disclosure_depends_on_delivery_and_permission(variant, authorized, expected):
    w = world(variant=variant, authorized=authorized)
    s, _ = phase.native_source_prefix(w.env)
    original = list(s.ledger)
    _, _, audit = w.branches(s, "SEND_RAW_TARGET")[0]
    assert audit["events"] and audit["inferred"] is expected
    assert audit["direct"] is False
    assert (audit["verdict"] == "VIOLATION") is expected
    assert s.ledger == original


def test_insufficient_horizon_and_ties_do_not_create_long_horizon_labels():
    w = world()
    e = Alignment(w, w.reference, "fixed", horizon=1)
    s = w.initial()
    a, b = e.evaluate(s, 1)
    assert a["reference"]["unknown"] == 1
    assert a["witness"] is None
    assert e.preference(a, b) is None
    assert e.preference(b, b) is None
    trace = e.visit(7)
    assert trace["unknown"] and not trace["stsr"] and not e.pairs


def test_repeated_prohibited_raw_fact_is_direct_even_without_new_inference():
    w = world()
    s = w.initial()
    s.delivered[w.env.spec.target].append(w.env.case_subject+"::"+FINAL)
    report = w.env._deliver(s, "SourceAgent", w.env.spec.target, FINAL, "protected association")
    assert not report["new_relations"] and not report["unlicensed_protected_new"]
    assert report["direct_violation"] and s.violation


def test_dpr_uses_real_failed_downstream_action_at_unchanged_observation():
    w = world()
    def choose(ob):
        return "SEND_RAW_TARGET" if ob["stage"] == "release" else w.reference(ob)
    e = Alignment(w, choose, "frozen")
    t = e.visit(7)
    assert t["inferred"] and not t["stsr"]
    repairs = [p for p in e.pairs.values() if p["source"] == "repair"]
    assert repairs
    for p in repairs:
        assert p["trace_id"] == t["id"] and p["observation_id"] == key(p["prompt"])
        step = next(h for h in t["path"] if h["state"] == p["state"])
        assert step["observation_id"] == p["observation_id"]
        assert step["action"] == p["rejected"] == "SEND_RAW_TARGET"
        assert p["actor"] == step["actor"]
    forced = Alignment(w, choose, "frozen")
    forced.visit(7, "reference_prefix", first=(lambda o:o["stage"] == "release", choose))
    assert not forced.failures and all(p["source"] != "repair" for p in forced.pairs.values())


def test_reference_witness_replays_to_actual_safe_task_commit():
    w = world()
    s, _ = phase.native_source_prefix(w.env)
    e = Alignment(w, w.reference, "fixed")
    rows = e.evaluate(s, e.horizon)
    for r in rows:
        if not r.get("witness"):
            continue
        current = s
        for index, h in enumerate(r["witness"]):
            assert h["before"] == e.sid(current)
            assert h["action"] == (r["action"] if index == 0 else w.reference(w.observe(current)))
            _, current, audit = w.branches(current, h["action"])[0]
            assert h["after"] == e.sid(current) and audit["verdict"] == "VALID"
        assert w.terminal(current)[:2] == (True, True)
        assert not w.omitted(current)
        break
    else:
        pytest.fail("No lawful task-completing witness")

