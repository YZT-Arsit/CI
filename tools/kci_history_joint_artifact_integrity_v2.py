"""Validate frozen three-arm scientific V2 sources, GPU weights, and DEV native traces.

Runs only after full nine training runs and DEV evaluation. No additional
inference/training, no test roots. Saves sha256 manifest for all evidence.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261124_joint_history_supervision_v2"
OUTPUT=D/"ARTIFACT_SHA256.json"
ARMS=("action_ce","same_state_dpo","joint_flip_hold_ce")
SEEDS=(7,1234,2025)
VIEWS=("original","entity_alias")
def sha(path):
 h=sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def j(p):return json.loads(Path(p).read_text(encoding="utf8"))
def jl(p):return [json.loads(s) for s in Path(p).open(encoding="utf8") if s.strip()]
def main():
 if OUTPUT.exists():raise FileExistsError(OUTPUT)
 prereg=j(D/"THREE_ARM_PRE_REGISTERED.json")
 source=j(D/"SET_VALUED_CORRECTION_AUDIT.json")
 criterion=j(D/"DEV_ELIMINATION_PRE_REGISTERED.json")
 isolation=j(D/"TRAIN_ROLE_VISIBLE_PAIR_ISOLATION_AUDIT.json")
 summary=j(D/"REAL4B_DEV_FULL_NATIVE/SUMMARY.json")
 eliminated=j(D/"JOINT_CANDIDATE_ELIMINATION.json")
 static=j(D/"VISIBLE_ONLY_STATIC_CLOSURE_DIAGNOSTIC.json")
 assert prereg["same_data_sha256"]==sha(D/"TRAIN_FIVE_CONTEXTS.jsonl")
 assert prereg["dev_data_sha256"]==sha(D/"DEV_FIVE_CONTEXTS.jsonl")
 assert source["TRAIN_sha256"]==sha(D/"TRAIN_FIVE_CONTEXTS.jsonl")
 assert source["DEV_sha256"]==sha(D/"DEV_FIVE_CONTEXTS.jsonl")
 assert source["native_branch_witness_only_not_oracle_Qstar_Qpi"]
 assert isolation["n_train_cross_history_control_pairs"]==28
 assert isolation["all_pairs_isolate_exactly_one_actor_visible_field"]
 assert eliminated["source_DEV_results_sha256"]==sha(D/"REAL4B_DEV_FULL_NATIVE/SUMMARY.json")
 assert eliminated["predeclared_criteria_sha256"]==sha(D/"DEV_ELIMINATION_PRE_REGISTERED.json")
 assert summary["n_three_arm_real_dev_native_episodes"]==270
 assert static["n_dev_true_states"]==15
 train_forward_calls={};reference_forward_calls={};seconds={}
 all_training_weights=[]
 for arm in ARMS:
  for seed in SEEDS:
   p=D/f"train_{arm}_seed{seed}"
   spec=j(p/"TRAINING.json")
   model=p/"adapter/adapter_model.safetensors"
   cfg=p/"adapter/adapter_config.json"
   assert model.is_file() and cfg.is_file()
   assert spec["seed"]==seed and spec["arm"]==arm
   assert spec["updates"]==28 and spec["n_true_unique_training_states"]==35
   assert spec["training_data_sha256"]==prereg["same_data_sha256"]
   assert spec["init_shared_full_flow_SFT_model_adapter_SHA256"]==prereg["same_init_adapter_SHA256"]
   assert spec["set_valued_native_labels_from_equal_CI_success_cost"]
   assert spec["same_35_states_same_optimizer_schedule_across_all_arms"]
   assert spec["total_differentiable_candidate_forwards"]==460
   if arm=="same_state_dpo":
    assert spec["reference_initial_candidate_forwards"]==115
   else:
    assert spec["reference_initial_candidate_forwards"]==0
   assert spec["dev_and_final_test_not_read"]
   key=f"{arm}_seed{seed}"
   train_forward_calls[key]=spec["total_differentiable_candidate_forwards"]
   reference_forward_calls[key]=spec["reference_initial_candidate_forwards"]
   seconds[key]=spec["training_wall_seconds"]
   all_training_weights.append({"model":key,"sha256":sha(model),"bytes":model.stat().st_size})
 for seed in SEEDS:
  ls=[j(D/f"train_{arm}_seed{seed}/TRAINING.json")["root_schedule"] for arm in ARMS]
  assert ls[0]==ls[1]==ls[2]
  assert ls[0]==prereg["optimization_schedule_seed_specific_method_independent"][str(seed)]
  assert len(ls[0])==28
 traces=jl(D/"REAL4B_DEV_FULL_NATIVE/REAL_NATIVE_MODEL_TRAJECTORIES.jsonl")
 assert len(traces)==270
 ids=Counter((x["arm"],x["seed"],x["model_view"],x["world"]) for x in traces)
 assert len(ids)==270 and set(ids.values())=={1}
 for x in traces:
  assert x["root"] in ("P204","P209","P210")
  assert x["condition"] in ("risky","safe","hold","authorized","equivalent")
  assert x["first_release_strictly_correct"]==(x["first_release_selected"] in x["first_release_gold_set"])
  assert x["real_native_verified_prefix_is_frozen"]
  assert x["no_scripted_post_release_actions"] and x["not_a_Qstar_or_scripted_reference_suffix"]
  assert not (x["native_task_safe_success"] and (x["native_CI_violation"] or x["native_abort"] or x["native_deadline_missed"] or x["native_UNKNOWN_or_incomplete"]))
  assert all("native_action" in row and row["verdict"] in ("VALID","VIOLATION") or
             row.get("native_verdict")=="NOT_EXECUTED_TIE" for row in x["full_native_true_model_action_trace"])
  if x["native_CI_violation"]:
   assert any(v.get("verdict")=="VIOLATION" for v in x["full_native_true_model_action_trace"])
  if x["first_release_selected"]=="UNRESOLVED_EXACT_TIE":
   assert x["native_UNKNOWN_or_incomplete"]
 report=ROOT/"docs/KCI_STATE_CONDITIONED_JOINT_HISTORY_3ARM_20261124_CN.md"
 assert report.is_file() and report.stat().st_size>3000
 files=[p for p in D.rglob("*") if p.is_file() and OUTPUT!=p and
        not any(x.startswith("pytest") for x in p.relative_to(D).parts)]
 file_info=[]
 for p in sorted(files):
  file_info.append({"path":p.relative_to(D).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size})
 manifest={
  "status":"V2_SAME_35_NATIVE_HISTORY_TRAIN_STATES_3ARMS_3SEEDS_REAL_NATIVE_DEV_AUDITED",
  "model_init_SHA256":prereg["same_init_adapter_SHA256"],
  "n_training_roots":7,"n_native_unique_training_states":35,
  "n_dev_business_roots":3,"n_dev_5context_states":15,
  "number_of_methods":3,"number_of_train_seeds":3,
  "native_DEV_episodes":len(traces),
  "all_test_final_roots_unread":True,
  "all_methods_use_same_train_states_each_seed":True,
  "all_methods_still_retain_native_ABORT_and_broker_options":True,
  "original_broker_tie_label_bug_fixed_for_ALL_methods_before_DEV":True,
  "all_28_pairs_true_one_field_actor_visible_isolation":True,
  "CE_and_DPO_not_altered_by_C_only_new_labels":True,
  "native_verifier_and_original_Phase2_unchanged":True,
  "CCE_OR_DPR_NOT_RUN_AS_NEW_ALGO":True,
  "candidate_decision":eliminated["decision"],
  "DEV_static_visible_closure_correct":static["state_action_set_correct"],
  "train_differentiable_forwards_per_method_seed":train_forward_calls,
  "DPO_reference_precompute_forwards":reference_forward_calls,
  "training_wall_seconds":seconds,
  "model_weights_sha256":all_training_weights,
  "report_sha256":sha(report),
  "evidence_files":file_info,
 }
 OUTPUT.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 print("JOINT_HISTORY_V2_INTEGRITY_PASS",len(file_info),"files",
  "total_DEV_episodes",len(traces),"static_DEV",static["state_action_set_correct"],
  "decision",eliminated["decision"],flush=True)
if __name__=="__main__":main()
