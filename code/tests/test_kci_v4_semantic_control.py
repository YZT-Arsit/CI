"""V4 fixed semantic interface and true-history-blind label tests."""
from pathlib import Path
import sys,json,hashlib
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_v4_semantic_action import (semantic_v4_candidate,canonical_v4_state_text,
                                    assert_views_position_invariant,FORMAT)
from kci_v4_semantic_train import fixed_schedule,load_training
D=ROOT/"artifacts/20261123_v4_semantic_control_v1"
def lines(n):
    return [json.loads(x) for x in (D/n).read_text(encoding="utf8").splitlines()]
def test_two_orientations_identical_semantics():
    a=lines("TRAIN_HISTORY_AWARE.jsonl")
    assert len(a)==9 and all(x["role"] and x["stage"]=="release" for x in a)
    for row in a:
        assert assert_views_position_invariant(row["prompt"],row["original_action_order"],
              row["swapped_prompt"],row["swapped_action_order"])
        assert all(hashlib.sha256(semantic_v4_candidate(
                  row["swapped_prompt"],row["swapped_action_order"],act).encode()).hexdigest()==
                   row["semantic_candidate_sha256"][act] for act in row["canonical_actions"])
        assert "Actually received:" in canonical_v4_state_text(row["prompt"],row["original_action_order"])
        assert "Candidate action under consideration:" in semantic_v4_candidate(row["prompt"],row["original_action_order"],"SEND_RAW_R")
        assert "ABORT" in semantic_v4_candidate(row["prompt"],row["original_action_order"],"ABORT")
def test_blind_only_changes_label_never_actor_history():
    aware=lines("TRAIN_HISTORY_AWARE.jsonl")
    blind=lines("TRAIN_HISTORY_BLIND.jsonl")
    assert len(aware)==len(blind)==9
    for a,b in zip(aware,blind):
        assert a["comparison_id"]==b["comparison_id"]
        assert a["prompt"]==b["prompt"]
        assert a["semantic_candidate_sha256"]==b["semantic_candidate_sha256"]
        assert a["true_history_aware_action"]==b["true_history_aware_action"]
        assert a["true_history_blind_action"]==b["true_history_blind_action"]
        assert b["chosen_action"]=="SEND_RAW_R"
        assert b["blind_native_justification"]["native_RAW_verdict"]=="VALID"
        assert b["blind_native_justification"]["raw_fixed_suffix_steps"]<b["blind_native_justification"]["verify_fixed_suffix_steps"]
def test_frozen_seed_schedule_and_equal_normalized_inputs():
    a=load_training(D/"TRAIN_HISTORY_AWARE.jsonl","aware_ce")
    b=load_training(D/"TRAIN_HISTORY_BLIND.jsonl","blind_ce")
    assert fixed_schedule(a,7,96)==fixed_schedule(b,7,96)
    assert len(set(x["comparison_id"] for x in a))==9
def test_dev_native_vs_entity_alias_differs_only_by_names():
    rows=lines("DEV_DUAL_VIEW_DG34_DG35.jsonl")
    assert len(rows)==10
    for r in rows:
        assert set(r["canonical_actions"])=={"SEND_RAW_R","VERIFY_PRIVATE_R","ABORT"}
        assert assert_views_position_invariant(r["native_prompt"],r["original_action_order"],
                r["native_swapped_prompt"],r["swapped_action_order"])
        assert r["aliased_prompt"]!=r["native_prompt"]
        assert "CASE-ALIAS-FOR-ROBUSTNESS" in r["aliased_prompt"]
        assert "Actually received:" in r["aliased_prompt"]
        assert r["root"] in ("DG34","DG35")
        assert r["true_native_action"]==("VERIFY_PRIVATE_R" if r["condition"]=="risky" else "SEND_RAW_R")
