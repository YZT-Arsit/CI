"""Exercise the matched-action guard on real, failed and successful baselines."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"kci"))
from v9_history_repair import cases, make_schedule


def native_example():
    def leaf(success=False,leak=False,abort=False):
        return dict(success=success,stsr=success and not leak,direct=False,inferred=leak,abort=abort)
    a=dict(x=leaf(True),y=leaf(),z=leaf(abort=True))
    b=dict(x=leaf(True,True),y=leaf(),z=leaf(abort=True))
    def branch(tag, outcomes):
        return [dict(policy_input=tag,outcomes=outcomes)]
    g=dict(id="r",source_root="theory",split="fit",valid_first={"a":True,"b":True},
           rejected="b",paths={"a":branch("delivered_a",a),"b":branch("delivered_b",b)})
    def prompt(h):
        return dict(prompt='Choose\n{"actor":"B","recipient_actual_deliveries":["'+h+'"]}\nExecutable actions:\nx\ny\nz')
    ps={t:prompt(t) for t in ("delivered_a","delivered_b")}
    actual=dict(input="delivered_a",action="y",**a["y"])
    baseline=dict(root_results=[dict(id="r",first="a",events={"a":[actual]})])
    return g,ps,baseline


def test_certified_repair_and_opposite_history_have_same_semantic_actions():
    g,ps,base=native_example()
    ev=cases([g],ps,base)
    assert ev["r"]["matched"]==["x"]
    schedule=make_schedule([g],ev,7)
    for d,c in zip(schedule["dpr"],schedule["paired_ci"]):
        assert c["pos"]==d["pos"]==[["delivered_a","x"]]
        assert c["neg"]==d["neg"]==[["delivered_a","y"]]
        assert c["common"]==[["delivered_b","y"]]
        assert c["common_neg"]==[["delivered_b","x"]]


@pytest.mark.parametrize("change",["no_violation","unsafe_negative","not_reached","illegal_first"])
def test_no_strict_dpr_guard_without_all_witnesses(change):
    g,ps,base=native_example()
    if change=="no_violation":g["paths"]["b"][0]["outcomes"]["x"]["inferred"]=False
    if change=="unsafe_negative":g["paths"]["b"][0]["outcomes"]["y"]["direct"]=True
    if change=="illegal_first":g["valid_first"]["a"]=False
    if change=="not_reached":
        root=base["root_results"][0];root["events"]["a"]=[dict(input="delivered_a",action="x",**g["paths"]["a"][0]["outcomes"]["x"])]
    assert cases([g],ps,base)["r"]["matched"]==[]


def test_context_change_beyond_the_receipts_invalidates_the_comparison():
    g,ps,base=native_example()
    ps["delivered_b"]["prompt"]=ps["delivered_b"]["prompt"].replace('"actor":"B"','"actor":"A"')
    with pytest.raises(AssertionError):cases([g],ps,base)

