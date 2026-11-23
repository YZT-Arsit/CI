"""Correct nonunique valid SOURCE actions before final matched study.

V1 retrospective functional bug: P206/P207 risky states allowed BOTH
VERIFY_PRIVATE_TARGET and SEND_RAW_BROKER with equal real native safe-success
and time cost. A single forced VERIFY gold would incorrectly penalize valid
broker. Preserve V1 artifacts, NEW V2 immutable data.
No dev model used and final_test never parsed; all three arms get identical
acceptable-action sets and exact same input states. No change to V4 truth.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/"artifacts/20261124_joint_history_supervision_v1"
V2=ROOT/"artifacts/20261124_joint_history_supervision_v2"
NATIVE=Path(r"D:\research\kci_data_v4_phase2_20261122\artifacts\20261122_kci_data_v4_phase2_full_v1\FULL_NATIVE_ACTION_TRAJECTORIES_EVAL_ONLY.jsonl")
VALIDROOTS={"P201","P202","P203","P204","P205","P206","P207","P208","P209","P210"}
PREFER=("SEND_RAW_TARGET","VERIFY_PRIVATE_TARGET","SEND_RAW_BROKER")
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).open(encoding="utf8") if x.strip()]
def native_by_world():
    out={}
    for line in NATIVE.open(encoding="utf8"):
        if not any(('"root_id":"'+r+'"') in line for r in VALIDROOTS):continue
        x=json.loads(line)
        assert x["root_id"] in VALIDROOTS and x["split"] in ("train","dev")
        assert x["world_id"] not in out
        out[x["world_id"]]=x
    assert len(out)==50
    return out
def accept(branches,actions):
    allowed=[a for a in PREFER if a in actions]
    assert "SEND_RAW_TARGET" in allowed and "VERIFY_PRIVATE_TARGET" in allowed
    options={}
    for act in allowed:
        detail=branches[act]
        suffix=detail["real_fixed_suffix"]
        options[act]={
            "immediate":detail["immediate_verdict"],
            "native_valid_complete_success":bool(suffix["success"] and not suffix["violation"]),
            "native_future_violation":bool(suffix["violation"]),
            "native_deadline":bool(suffix["deadline_missed"]),
            "native_steps":int(suffix["total_steps"]),
            "evidence_kind":"native_fixed_policy_witness_not_Qstar_or_Qpi"}
    lawful=[a for a in allowed if options[a]["immediate"]=="VALID" and not options[a]["native_future_violation"]]
    assert lawful
    winners=[a for a in lawful if options[a]["native_valid_complete_success"]]
    if winners:
        best=min(options[a]["native_steps"] for a in winners)
        acceptable=[a for a in winners if options[a]["native_steps"]==best]
        reason="same_native_state_CI_safe_full_business_success_min_actual_witness_cost"
    else:
        acceptable=lawful[:]
        reason="no_verified_CI_safe_task_success_in_available_frozen_suffixes_allow_all_nonviolating_immediate_options"
    assert acceptable and all(a in actions for a in acceptable)
    assert "ABORT" in actions
    rejected=[a for a in allowed if a not in acceptable]
    assert rejected
    return acceptable,rejected,options,reason
def main():
    if V2.exists():raise FileExistsError(V2)
    source=native_by_world()
    v1rows={z:[x for x in read(V1/(z+"_FIVE_CONTEXTS.jsonl"))]
            for z in ("TRAIN","DEV")}
    assert len(v1rows["TRAIN"])==35 and len(v1rows["DEV"])==15
    V2.mkdir(parents=True)
    stats=Counter()
    for split,rows in v1rows.items():
        result=[]
        for r in rows:
            orig=source[r["world"]]
            assert orig["root_id"]==r["root"] and orig["split"]==r["split"]
            acceptable,rejected,branches,reason=accept(orig["native_action_branches"],r["available_native_actions"])
            # Preserve original single-gold as audit-only historical bug.
            entry={**r,
                "v1_single_gold_no_longer_teacher":r["native_gold"],
                "acceptable_native_actions":acceptable,
                "nonpreferred_same_state_comparators":rejected,
                "acceptable_native_action_rule":reason,
                "all_executable_actions_original":r["available_native_actions"],
                "native_branch_evidence_non_actor":branches,
                "native_gold":"SET_VALUED_REPLACED_IN_V2",
                "same_state_rejected":"SET_VALUED_REPLACED_IN_V2",
                "not_a_new_oracle_label_or_CCE_Qpi":True}
            assert all(a not in rejected for a in acceptable)
            result.append(entry)
            stats[(split,r["condition"],tuple(acceptable))]+=1
            if r["root"] in ("P206","P207","P210") and r["condition"]=="risky":
                assert set(acceptable)=={"VERIFY_PRIVATE_TARGET","SEND_RAW_BROKER"}
            elif r["condition"]=="risky":
                assert acceptable==["VERIFY_PRIVATE_TARGET"]
            else:
                assert acceptable==["SEND_RAW_TARGET"],r["world"]
        p=V2/(split+"_FIVE_CONTEXTS.jsonl")
        p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in result),encoding="utf8")
        assert len(result)==(35 if split=="TRAIN" else 15)
        # Relation evidence unchanged.
        (V2/(split+"_RELATION_PAIRS.jsonl")).write_bytes((V1/(split+"_RELATION_PAIRS.jsonl")).read_bytes())
    report={
      "status":"V2_FROZEN_FUNCTIONAL_LABEL_SET_CORRECTION",
      "source_original_V4_native_data_readonly_SHA256":sha(NATIVE),
      "old_v1_frozen_data_preserved":True,
      "reason":"P206/P207 risky VERIFY and licensed broker both have equal safe full business outcomes and cost",
      "source_model_train_contexts_unchanged":True,
      "source_model_dev_contexts_unchanged":True,
      "no_extra_history_or_semantic_views_for_any_arm":True,
      "native_branch_witness_only_not_oracle_Qstar_Qpi":True,
      "TRAIN_rows":35,"DEV_rows":15,
      "corrected_acceptable_action_sets_summary":{str(k):v for k,v in stats.items()},
      "TRAIN_sha256":sha(V2/"TRAIN_FIVE_CONTEXTS.jsonl"),
      "DEV_sha256":sha(V2/"DEV_FIVE_CONTEXTS.jsonl"),
      "TRAIN_relation_evidence_sha256":sha(V2/"TRAIN_RELATION_PAIRS.jsonl"),
      "DEV_relation_evidence_sha256":sha(V2/"DEV_RELATION_PAIRS.jsonl"),
      "original_v1_train_sha256":sha(V1/"TRAIN_FIVE_CONTEXTS.jsonl"),
      "DEV_not_loaded_into_training":True,
      "no_dev_model_outcomes_or_final_test_used":True,
      "no_data_source_corruption":True}
    (V2/"SET_VALUED_CORRECTION_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("JOINT_V2_TRUE_ACCEPTABLE_SET_FROZEN",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
