"""Independent method comparison audit: root-clustered descriptive, no pseudo-replication.

The three supervisors have exactly identical labels on all seven TRAIN A
states. Never infer algorithm superiority from GPU non-determinism.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/"artifacts/20261124_cce_method_comparison_v1"
SFT=ROOT/"artifacts/20261124_cce_downstream_sft_v1"
CCE=ROOT/"artifacts/20261124_cce_phase2_post_sft_v1"
METHODS=("downstream_sft_untrained_A","action_ce","terminal","CCE")
ROOTS=("P204","P209","P210")
def jl(p):return [json.loads(x) for x in p.open(encoding="utf8") if x.strip()]
def read(p):return json.loads(p.read_text(encoding="utf8"))
def sha(p):
    d=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):d.update(chunk)
    return d.hexdigest()
def main():
    xs=jl(B/"DEV_3ARM_REAL4B/REAL_UNSEEN_DEV_NATIVE_EPISODES.jsonl")
    assert len(xs)==288
    s=read(B/"DEV_3ARM_REAL4B/SUMMARY.json")
    assert s["n_complete_native_4B_dev_episodes"]==len(xs)
    grouped=defaultdict(dict)
    for r in xs:
        key=(r["root"],r["condition"],r["seed"])
        assert r["model"] not in grouped[key]
        assert r["root"] in ROOTS
        assert r["all_steps_actual_native_transitions"] and r["real_4B_no_Qstar_in_prompt"]
        grouped[key][r["model"]]=r
    assert len(grouped)==72 and all(set(v)==set(METHODS) for v in grouped.values())
    audit={"protocol":"one seed, three independently held out business roots only",
      "n_true_business_roots":3,
      "n_related_condition_seed_configurations":72,
      "one_shared_model_different_adapters":True,
      "train_signal_label_identity_confirmed_for_7_roots":True,
      "full_native_outcomes_not_classifier_accuracy":True,
      "DEV_mechanism_overlap_known":True,
      "not_claiming_significance_from_72_correlated_episodes":True,
      "all_models_from_same_SFT_adapter_start":True,
      "per_root_method_success":{},
      "per_root_method_future_CI":{},
      "paired_state_differences":{},
      "model_action_sequences_agree_by_pair":{},
      "terminal_and_Action_CE_train_target_equal_7_of_7":True}
    for rt in ROOTS:
        audit["per_root_method_success"][rt]={m:sum(z["task_CI_safe_success"] for z in xs if z["root"]==rt and z["model"]==m) for m in METHODS}
        audit["per_root_method_future_CI"][rt]={m:sum(z["CI_violation"] for z in xs if z["root"]==rt and z["model"]==m) for m in METHODS}
    for method in METHODS:
        if method=="downstream_sft_untrained_A":continue
        wins=Counter()
        action_agree=0;sequence_agree=0
        for key,g in grouped.items():
            a=g[method];z=g["downstream_sft_untrained_A"]
            wins["safe_success_improves"]+=bool(a["task_CI_safe_success"] and not z["task_CI_safe_success"])
            wins["safe_success_lost"]+=bool(z["task_CI_safe_success"] and not a["task_CI_safe_success"])
            wins["new_future_CI_violation"]+=bool(a["CI_violation"] and not z["CI_violation"])
            wins["avoided_old_future_CI_violation"]+=bool(z["CI_violation"] and not a["CI_violation"])
            action_agree+=int(a["A_choice"]==z["A_choice"])
            sequence_agree+=int([ev["native_action"] for ev in a["native_stepwise_trace"]]==
                                 [ev["native_action"] for ev in z["native_stepwise_trace"]])
        audit["paired_state_differences"][method]=dict(wins)
        audit["model_action_sequences_agree_by_pair"][method]={"upstream_A_choice_matches_base":action_agree,
                                                                "all_actions_match_base":sequence_agree}
    for method in ("terminal","CCE"):
        group=Counter()
        for key,rr in grouped.items():
            a=rr[method];b=rr["action_ce"]
            group["A_action_same_as_CE"]+=a["A_choice"]==b["A_choice"]
            group["full_native_sequence_same_as_CE"]+=(
                [e["native_action"] for e in a["native_stepwise_trace"]]==
                [e["native_action"] for e in b["native_stepwise_trace"]])
            group["same_safe_task_outcome_as_CE"]+=(a["task_CI_safe_success"]==b["task_CI_safe_success"])
            group["same_CI_violation_as_CE"]+=(a["CI_violation"]==b["CI_violation"])
            group["CE_wins_safe_task"]+=int(b["task_CI_safe_success"] and not a["task_CI_safe_success"])
            group["arm_wins_safe_task"]+=int(a["task_CI_safe_success"] and not b["task_CI_safe_success"])
        audit["paired_state_differences"][method+"_vs_action_ce"]=dict(group)
    t={}
    schedule=set()
    for arm in ("action_ce","terminal","CCE"):
        row=read(B/f"train_{arm}_seed7/TRAINING.json")
        assert row["method"]==arm and row["steps"]==48
        assert row["all_three_teacher_labels_identical_across_7_states"]
        assert row["n_training_states"]==7
        assert row["ABORT_kept_in_model_action_menu"]
        assert row["initial_model_SFT_adapter_sha256"]==sha(SFT/"shared_native_sft_seed7/adapter/adapter_model.safetensors")
        assert row["source_teacher_file_sha256"]==sha(B/"MATCHED_SAME_A_STATE_THREE_TEACHERS_TRAIN_ONLY.jsonl")
        assert row["n_candidate_forward_calls"]==48*3
        schedule.add(row["schedule_sha256"])
        p=B/f"train_{arm}_seed7/adapter/adapter_model.safetensors"
        assert p.exists() and p.stat().st_size==11815504
        assert sha(p)==s["model_checkpoint_sha256"][arm]
        t[arm]={"steps":row["steps"],"n_training_states":row["n_training_states"],
                "n_candidate_forward_calls":row["n_candidate_forward_calls"],
                "wall_training_seconds":row["wall_training_seconds"],
                "source_weights_sha256":sha(p)}
    assert len(schedule)==1
    audit["matched_A_training_3arms"]=t
    audit["matched_schedule_sha256"]=schedule.pop()
    audit["DEV_native_trace_SHA256"]=sha(B/"DEV_3ARM_REAL4B/REAL_UNSEEN_DEV_NATIVE_EPISODES.jsonl")
    audit["DEV_summary_SHA256"]=sha(B/"DEV_3ARM_REAL4B/SUMMARY.json")
    assert not (B/"COMPARISON_AUDIT_v2.json").exists()
    (B/"COMPARISON_AUDIT_v2.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_3ARM_DEV_AUDIT",json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
