"""V4 train-root controls are native-grounded and cannot leak dev roots."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_v4_train_root_controls_eval import stats
D=ROOT/"artifacts/20261123_v4_paired_repair_v1/TRAIN_ROOT_HOLD_ALIAS_EVAL_ONLY.jsonl"
def test_root_train_control_freeze_counts():
    rows=[json.loads(s) for s in D.read_text(encoding="utf8").splitlines()]
    assert len(rows)==15 and {r["root"] for r in rows}=={"DG31","DG32","DG33"}
    assert sum(r["train_source_state"] for r in rows)==9
    assert sum(not r["train_source_state"] for r in rows)==6
    for r in rows:
        assert r["NO_DG36_data_and_no_dev_labels"]
        assert r["native_gold_action"]==("VERIFY_PRIVATE_R" if r["condition"]=="risky" else "SEND_RAW_R")
def test_true_strict_pair_metric_denominators_for_all_5_states():
    rows=[json.loads(s) for s in D.read_text(encoding="utf8").splitlines()]
    summary=stats([{"root":r["root"],"condition":r["condition"],
        "gold":r["native_gold_action"],
        "prediction":r["native_gold_action"],
        "strict":True,"seen_train":r["train_source_state"],
        "raw_minus_verify":-1 if r["condition"]=="risky" else 1}
        for r in rows])
    assert summary["strictly_correct"]==15
    assert all(x["correct_roots"]==3 and x["denominator"]==3 for x in summary["pairs"].values())
