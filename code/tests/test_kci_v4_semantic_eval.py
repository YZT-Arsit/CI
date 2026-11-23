"""V4 semantic scorer and dual-view native metrics regression."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_v4_semantic_eval import source_states,decision_model_metrics
from kci_v4_semantic_action import semantic_v4_candidate,assert_views_position_invariant
D=ROOT/"artifacts/20261123_v4_semantic_control_v1"
def read(name):
    return [json.loads(line) for line in (D/name).read_text(encoding="utf8").splitlines()]
def test_semantic_permutation_and_double_view():
    a=read("TRAIN_HISTORY_AWARE.jsonl");dev=read("DEV_DUAL_VIEW_DG34_DG35.jsonl")
    states=source_states(a,dev)
    assert len(states)==19 and len(set(s["root"] for s in states))==5
    assert all(assert_views_position_invariant(s["prompt_native"],s["options"],s["swapped_prompt"],s["swapped_options"]) for s in states)
    for s in states:
        actions=sorted(s["options"])
        assert "ABORT" in actions
        for act in actions:
            assert "Answer A if this candidate should be preferred" in semantic_v4_candidate(s["prompt_native"],s["options"],act)
def test_perfect_pair_metrics_must_equal_3_train_2_dev():
    states=source_states(read("TRAIN_HISTORY_AWARE.jsonl"),read("DEV_DUAL_VIEW_DG34_DG35.jsonl"))
    for split,n,nroots in (("train",9,3),("dev",10,2)):
        xs=[]
        for state in states:
            if state["split"]!=split:continue
            truth=state["gold"]
            xs.append({"root":state["root"],"condition":state["condition"],"selected":truth,
                "gold":truth,"P":{a:(0.98 if a==truth else 0.01) for a in state["options"]},
                "native_outcomes":state["native_outcomes"]})
        res=decision_model_metrics(xs)
        assert res["state_actions_correct"]==n
        assert res["pair_metrics"]["Flip"]["both_correct"]==nroots
        assert res["pair_metrics"]["Authorized"]["both_correct"]==nroots
        assert res["raw_on_risky"]==0
        assert res["native_fixed_suffix_CI_violations"]==0
        if split=="dev":
            assert res["pair_metrics"]["Hold"]["both_correct"]==2
            assert res["pair_metrics"]["EquivalentHistoryOrder"]["both_correct"]==2
            assert res["authorized_raw"]==2
def test_blind_is_distinct_from_true_aware_labels():
    a=read("TRAIN_HISTORY_AWARE.jsonl");b=read("TRAIN_HISTORY_BLIND.jsonl")
    assert len(a)==len(b)==9
    assert sum(x["chosen_action"]=="VERIFY_PRIVATE_R" for x in a)==3
    assert sum(x["chosen_action"]=="VERIFY_PRIVATE_R" for x in b)==0
    assert all(x["prompt"]==y["prompt"] for x,y in zip(a,b))
