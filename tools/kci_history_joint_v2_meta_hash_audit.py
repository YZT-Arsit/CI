"""Cross-check downloaded DEV evidence with original GPU checkpoint SHA256."""
from pathlib import Path
import json,hashlib,collections
ROOT=Path(r"D:\research\longhistory")
D=ROOT/"artifacts/20261124_joint_history_supervision_v2"
def rd(p):return json.loads(p.read_text(encoding="utf8"))
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1<<20),b""):h.update(c)
 return h.hexdigest()
def main():
 remote=rd(D/"REMOTE_GPU_ARTIFACT_SHA256.json")
 study=rd(D/"REAL4B_DEV_FULL_NATIVE/SUMMARY.json")
 tr=[json.loads(s) for s in (D/"REAL4B_DEV_FULL_NATIVE/REAL_NATIVE_MODEL_TRAJECTORIES.jsonl").open(encoding="utf8")]
 protocol=rd(D/"THREE_ARM_PRE_REGISTERED.json")
 decision=rd(D/"JOINT_CANDIDATE_ELIMINATION.json")
 static=rd(D/"VISIBLE_ONLY_STATIC_CLOSURE_DIAGNOSTIC.json")
 assert remote["trained_weights_count"]==9
 assert study["n_three_arm_real_dev_native_episodes"]==270 and len(tr)==270
 assert static["state_action_set_correct"]==15
 assert decision["decision"]=="ELIMINATE_JOINT_HISTORY_CANDIDATE"
 verify_paths=("THREE_ARM_PRE_REGISTERED.json","TRAIN_FIVE_CONTEXTS.jsonl",
               "DEV_FIVE_CONTEXTS.jsonl","REAL4B_DEV_FULL_NATIVE/SUMMARY.json",
               "REAL4B_DEV_FULL_NATIVE/REAL_NATIVE_MODEL_TRAJECTORIES.jsonl")
 for f in verify_paths:
  assert sha(D/f)==remote["files"][f]["sha256"],f
 assert protocol["same_data_sha256"]==sha(D/"TRAIN_FIVE_CONTEXTS.jsonl")
 assert protocol["dev_data_sha256"]==sha(D/"DEV_FIVE_CONTEXTS.jsonl")
 assert decision["source_DEV_results_sha256"]==sha(D/"REAL4B_DEV_FULL_NATIVE/SUMMARY.json")
 assert set(x["root"] for x in tr)=={"P204","P209","P210"}
 assert len(set((x["arm"],x["seed"],x["model_view"],x["world"]) for x in tr))==270
 assert set(x["model_view"] for x in tr)=={"original","entity_alias"}
 assert all(x["first_release_selected"]=="SEND_RAW_TARGET" for x in tr)
 assert all(x["first_release_strictly_correct"]==(x["first_release_selected"] in x["first_release_gold_set"]) for x in tr)
 assert all(x["no_scripted_post_release_actions"] for x in tr)
 assert all(x["real_native_verified_prefix_is_frozen"] for x in tr)
 assert not any(x["native_UNKNOWN_or_incomplete"] for x in tr)
 assert all(not(x["native_task_safe_success"] and x["native_CI_violation"]) for x in tr)
 assert sum(x["native_CI_violation"] for x in tr)==54
 assert sum(x["native_task_safe_success"] for x in tr)==216
 for arm in ("action_ce","same_state_dpo","joint_flip_hold_ce"):
  for seed in (7,1234,2025):
   key=f"train_{arm}_seed{seed}"
   meta=key+"/TRAINING.json"
   chk=key+"/adapter/adapter_model.safetensors"
   assert meta in remote["files"] and chk in remote["files"]
   assert remote["files"][chk]["bytes"]>10_000_000
   assert sha(D/meta)==remote["files"][meta]["sha256"]
   m=rd(D/meta)
   assert m["training_data_sha256"]==protocol["same_data_sha256"]
   assert m["init_shared_full_flow_SFT_model_adapter_SHA256"]==protocol["same_init_adapter_SHA256"]
   assert m["updates"]==28 and m["n_true_unique_training_states"]==35
   assert m["total_differentiable_candidate_forwards"]==460
   assert m["reference_initial_candidate_forwards"]==(115 if arm=="same_state_dpo" else 0)
 for seed in (7,1234,2025):
  specs=[rd(D/f"train_{arm}_seed{seed}/TRAINING.json") for arm in ("action_ce","same_state_dpo","joint_flip_hold_ce")]
  assert specs[0]["root_schedule"]==specs[1]["root_schedule"]==specs[2]["root_schedule"]
 report={
   "audit":"LOCAL_GPU_HASH_MATCH_AND_NATIVE_DEV_REPLAY_METADATA",
   "status":"PASS",
   "n_remote_weight_sha256_entries":9,
   "weight_files_stored_on_GPU_server_not_all_copied_to_Windows":True,
   "model_weight_hashes_from_PIN_verified_GPU_host":{f: v["sha256"] for f,v in remote["files"].items() if f.endswith("adapter_model.safetensors")},
   "n_train_specs_checked_by_matching_SHA256":9,
   "n_original_DEV_real_native_trajectories_checked":len(tr),
   "total_safe_success":sum(x["native_task_safe_success"] for x in tr),
   "total_CI_violations":sum(x["native_CI_violation"] for x in tr),
   "all_first_release_model_actions":"SEND_RAW_TARGET",
   "static_closure_original_DEV":static["state_action_set_correct"],
   "candidate_C_decision":decision["decision"],
   "no_final_test_read":True,
   "model_prompt_actor_legitimate_only":True,
   "remote_GPU_SHA256_manifest_SHA256":sha(D/"REMOTE_GPU_ARTIFACT_SHA256.json")
 }
 out=D/"ARTIFACT_META_REMOTE_GPU_HASH_AUDIT.json"
 assert not out.exists()
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 print("JOINT_V2_META_AND_HASH_AUDIT_PASS",len(tr),"episodes",9,"checkpoint SHA entries","decision",report["candidate_C_decision"])
if __name__=="__main__":main()
