"""No-dev CCE joint history native evaluation harness smoke."""
from pathlib import Path
from collections import defaultdict
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_history_joint_dev_real4b_eval_v2 import build_env,modified_candidate_prompt
D=ROOT/"artifacts/20261124_joint_history_supervision_v2/TRAIN_FIVE_CONTEXTS.jsonl"
def train_rows():return [json.loads(s) for s in D.open(encoding="utf8")]
def test_all_train_release_worlds_replay_exact_original_native_hash():
    r=train_rows()
    assert len(r)==35
    for x in r:
        env,state=build_env(x)
        assert state.stage=="release"
        assert set(x["acceptable_native_actions"])<=set(env.options(state))
def test_alias_semantics_uses_same_legitimate_input_options():
    r=train_rows()
    for x in (r[0],r[14],r[27]):
        ob=x["actor_visible_observation"]
        for a in ob["legal_actions"]:
            standard=modified_candidate_prompt(ob,a,"original")
            alias=modified_candidate_prompt(ob,a,"entity_alias")
            assert standard!=alias
            assert "Candidate native action: "+a in standard
            assert "Candidate native action: "+a in alias
            assert "Candidate native action: "+a in alias
            assert "ABORT" in alias
            assert "Delivered history:" in alias
def test_allowed_risky_broker_not_penalized_by_gold_set():
    r=train_rows()
    risk={x["root"]:x for x in r if x["condition"]=="risky"}
    for root in ("P206","P207"):
        assert set(risk[root]["acceptable_native_actions"])=={
            "SEND_RAW_BROKER","VERIFY_PRIVATE_TARGET"}
