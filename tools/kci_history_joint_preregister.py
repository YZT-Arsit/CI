"""Scientific frozen protocol before any shared-4B optimizer updates."""
from pathlib import Path
import json
from hashlib import sha256
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_history_joint_three_arm_train import (
    DATA_ROOT,CASES,ARMS,SEEDS,STEPS,LR,BETA,L_FLIP,L_AUTH,L_HOLD,L_EQUIV,sha,
    get_data,matched_root_schedule)
OUTPUT=DATA_ROOT/"THREE_ARM_PRE_REGISTERED.json"
START=ROOT/"artifacts/20261124_cce_downstream_sft_v1/shared_native_sft_seed7/adapter/adapter_model.safetensors"
def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    data=get_data(DATA_ROOT/"TRAIN_FIVE_CONTEXTS.jsonl")
    assert START.is_file()
    r={
      "status":"FROZEN_EVALUATION_BEFORE_TRAINING",
      "scientific_question":"old history contrast vs explicit joint Flip/Hold/authorized does it beat matched CE and same-state DPO?",
      "prior_existing_implementation":"tools/kci_v4_history_contrast_train.py",
      "not_claiming_new_algorithm":True,
      "dataset_version":"Phase2 frozen TRAIN and DEV native true paired histories",
      "train_rows":35,"train_independent_roots":7,"dev_rows":15,"dev_independent_roots":3,
      "root_split_train":list(data),
      "root_split_dev":["P204","P209","P210"],
      "each_group_context_conditions":list(CASES),
      "pair_relations":["history_flip","unrelated_hold","authorization_exception","equivalent_history"],
      "three_train_methods":list(ARMS),
      "identical_native_actor_observation_in_3_arms":True,
      "no_extra_contexts_or_new_swapped_option_training_for_C":True,
      "same_init_adapter_SHA256":sha(START),
      "same_model":"Qwen3-4B-Instruct-2507-ms / 4bit NF4 / BF16 / PEFT LoRA r8 alpha16 q,v",
      "same_one_shared_adapter_all_roles":True,
      "same_data_sha256":sha(DATA_ROOT/"TRAIN_FIVE_CONTEXTS.jsonl"),
      "dev_data_sha256":sha(DATA_ROOT/"DEV_FIVE_CONTEXTS.jsonl"),
      "same_candidate_prompt":"existing model_candidate_prompt: position free canonical native action names, logits ACCEPT-REJECT",
      "init_adapter_not_modified":True,
      "state_actions":["ABORT","SEND_RAW_TARGET","VERIFY_PRIVATE_TARGET"],
      "all_three_arms_keep_ABORT":True,
      "all_three_arms_have_same_root_stage_observation_schedule":True,
      "optimization_schedule_seed_specific_method_independent":{str(s):matched_root_schedule(data,s) for s in SEEDS},
      "seeds":list(SEEDS),"steps":STEPS,"state_rows_per_step":5,"seed":None,
      "lr":LR,"optimizer":"AdamW","gradient_clip":1.,
      "DPO_reference":"precomputed same SFT initialized adapter before any updates",
      "DPO_beta":BETA,
      "candidate_joint_old_flip_context_gap":True,
      "candidate_joint_flip_weight":L_FLIP,"candidate_joint_auth_weight":L_AUTH,
      "candidate_joint_hold_symKL_weight":L_HOLD,"candidate_joint_equiv_symKL_weight":L_EQUIV,
      "negative_label_source":"same-state actual native RAW vs VERIFY immediate verdict and fixed native witness; never cross-state DPO",
      "caveat_risky_tight_VERIFY_may_fail_task_deadline":True,
      "ties_do_not_count_as_correct":True,
      "gold_competitor_strict_margins_required":True,
      "eval_views":["original","entity_alias"],
      "eval_pair_metrics":["Flip_both_correct","Hold_both_correct","Auth_both_correct","Equiv_both_correct"],
      "eval_native_metrics":["valid_RAW_legal","unsafe_RAW_violations","CI_safe_task_success","abort","timeout","successful_steps"],
      "native_evaluation":"from actual release snapshot, frozen one shared 4B chooses every subsequent action in real native Phase2Env",
      "native_model_decoding":"greedy strict-ties unresolved; no reward/CI native oracle in actor prompt",
      "DEV_used_only_after_all_hyperparameters_frozen":True,
      "reject_C_if_not_stably_better_than_CE_on_unseen_roots":True,
      "reject_C_if_Flip_gain_compensated_by_loss_of_legal_RAW_or_CI_safe_task_success":True,
      "do_not_claim_independence_from_old_history_contrast":True,
      "do_not_touch_sealed_final_test":True,
      "no_CCE_or_DPR_training":True}
    OUTPUT.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("HISTORY_JOINT_PRE_REG",json.dumps({"seeds":list(SEEDS),"steps":STEPS,
     "train":35,"dev":15,"three_arms":list(ARMS),"init_SHA256":r["same_init_adapter_SHA256"],
     "contract_SHA256":sha(OUTPUT)},ensure_ascii=False))
if __name__=="__main__":main()
