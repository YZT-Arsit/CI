"""V4 original dual-order native outcome metric contract tests, no LLM required."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_v4_train_4b import read_groups,ACTION_INSTRUCTION
from kci_v4_eval_4b import build_states,state_decisions,metrics_for
BASE=ROOT/"artifacts/20261123_v4_release_control_v1"
def read(name):
    return [json.loads(x) for x in (BASE/name).read_text(encoding="utf8").splitlines()]
def states():
    return build_states(read_groups(BASE/"TRAIN_RELEASE_GROUPS.jsonl",True),
                       read("DEV_NATIVE_STATES_DG34_DG35.jsonl"))
def test_train_and_dev_native_gold():
    out=states()
    assert len(out)==19
    train=[x for x in out if x["split"]=="train"]
    dev=[x for x in out if x["split"]=="dev"]
    assert len(train)==9 and len(dev)==10
    assert sum(x["native_ground_truth"]=="VERIFY_PRIVATE_R" for x in train)==3
    assert sum(x["native_ground_truth"]=="SEND_RAW_R" for x in train)==6
    assert sum(x["native_ground_truth"]=="VERIFY_PRIVATE_R" for x in dev)==2
    assert sum(x["native_ground_truth"]=="SEND_RAW_R" for x in dev)==8
    assert all("DG36" not in x["world"] for x in out)
def test_pair_metric_counts_when_every_action_is_correct():
    all_=states()
    docs=[]
    for s in all_:
        class OracleLikeForUnitTestOnly:
            def distribution(self,prompt,actions,method):
                correct=s["native_ground_truth"]
                return {a:(0.98 if a==correct else 0.01) for a in actions}
        r=state_decisions(s,OracleLikeForUnitTestOnly())
        r["_source_state"]=s
        docs.append(r)
    for split,group_count,roots in (("train",9,3),("dev",10,2)):
        rows=[r for r in docs if r["split"]==split]
        metric=metrics_for(rows,"frozen_base")
        assert metric["normal"]["native_gold_action_correct"]==group_count
        assert metric["swapped"]["native_gold_action_correct"]==group_count
        assert metric["position_flip_states"]==0
        assert metric["paired_results"]["history_flip"]["correct_both_option_orders"]==roots
        assert metric["paired_results"]["authorization"]["correct_both_option_orders"]==roots
        assert metric["normal"]["native_CI_violations"]==0
        assert metric["normal"]["completed_extra_steps_sum"]==0
        assert metric["normal"]["safe_task_completion"]==group_count
        if split=="dev":
            assert metric["paired_results"]["irrelevant_hold"]["correct_both_option_orders"]==2
            assert metric["paired_results"]["equivalent_order"]["correct_both_option_orders"]==2
def test_constant_instruction_and_no_implicit_swap_augmentation():
    assert ACTION_INSTRUCTION
    groups=read_groups(BASE/"TRAIN_RELEASE_GROUPS.jsonl",True)
    assert all(x["n_original_records"]==2 for x in groups)
    assert sum(len(x["paired_option_records"]) for x in groups)==18
