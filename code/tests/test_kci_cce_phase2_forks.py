"""Smoke frozen-4B CCE Phase2 native fork machinery without GPU."""
from pathlib import Path
import sys,json,random
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_cce_phase2_frozen4b_rollout import one_trajectory,model_candidate_prompt
from kci.v4_phase2_workflows import Phase2Env,FlowState
from kci_cce_phase2_freeze_A import reconstruct
D=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1/FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl"
def row():
    return json.loads(D.read_text(encoding="utf8").splitlines()[0])
class FixedNativeLicensedPolicy:
    def get_distribution(self,ob):
        options=sorted(ob["legal_actions"])
        choices=(["OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL",
                  "SEND_RAW_TARGET","VERIFY_PRIVATE_TARGET",
                  "COMMIT_RESULT","SEND_CERT_TARGET",
                  "BROKER_FORWARD_CERT_TARGET"]+
                [x for x in options if x.startswith("DO:")]+
                [x for x in options if x.startswith("READ:")])
        action=next(x for x in choices if x in options)
        return {"distribution":{x:float(x==action) for x in options},
            "action_scores":{x:float(x==action) for x in options},
            "canonical_candidate_prompt_sha256":"not_a_real_4B_model"}
def test_frozen_same_A_state_two_actual_legal_deliveries():
    r=row()
    assert r["root"]=="P201" and r["condition"]=="risky"
    assert {x for x in r["A_valid_candidates"]}=={"A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"}
    assert len(set(v["post_A_native_hash"] for v in r["A_valid_candidates"].values()))==2
    for a,v in r["A_valid_candidates"].items():
        assert v["A_verdict"]=="VALID" and v["knowledge_delta"]
def test_tool_prefix_native_4B_continuation_runs_to_terminal():
    r=row()
    for a in ("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"):
        x=one_trajectory(r,a,7,FixedNativeLicensedPolicy(),"CPU_TOOL_SMOKE_ONLY")
        assert x["A_immediate_verdict"]=="VALID"
        assert x["start_A_state_hash"]==r["A_pre_state_sha256"]
        assert x["done"] and not x["unknown_or_unfinished"]
        assert x["native_transition_trace"]
        assert all(i["native_verdict"] in ("VALID","VIOLATION") for i in x["native_transition_trace"])
        if a=="A_SEND_PACKAGE_B":
            assert x["ci_safe_success"]  # native tight P201, safe package
