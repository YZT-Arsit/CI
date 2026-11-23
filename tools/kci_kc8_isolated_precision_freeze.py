"""Frozen compatible HF-only reload, no effect on data/labels or Qpi results."""
from pathlib import Path
from hashlib import sha256
import json
R=Path(r"D:\research\longhistory")
D=R/"artifacts/20261123_kc8_4b_policy_signal_v1"
def h(p):return sha256(Path(p).read_bytes()).hexdigest()
def main():
    path=D/"PRECISION_RUNTIME_ISOLATED_PATCH.json"
    if path.exists():raise FileExistsError(path)
    info={
      "status":"ISOLATED_RUNTIME_PRECISION_FIX_BEFORE_ANY_MODEL_KC8_TRAJECTORY",
      "source_frozen_protocol_sha256":h(D/"FROZEN_PROTOCOL.json"),
      "source_native_snapshot_sha256":h(D/"KC8_NATIVE_CODE_SNAPSHOT_TRAIN_ONLY.tar.gz"),
      "audited_HF_policy_sha256":h(D/"MATCHED_HF_ACTION_POLICY_PRECISION_V2.py"),
      "audited_HF_policy_source":"main code/kci/agent_rl/hf_policy.py; previously audited training vs inference 4bit calibration",
      "must_use_quantization":"4bit NF4 BF16 + PEFT prepare_model_for_kbit_training before PeftModel.from_pretrained",
      "functional_reason":"shared remote GPU agent_rl/hf_policy.py is older version without kbit_inference_precision kwarg",
      "original_failure_model_forward_count":0,
      "original_failure_model_trajectories":0,
      "old_failure_log":"artifacts/20261123_kc8_4b_policy_signal_v1/REAL4B_TRAIN_FROZEN.stdout.log on remote",
      "original_phase2_GPU_models_and_scripts_untouched":True,
      "data_AI_worktree_unchanged":True,
      "no_change_to_pre_reg_root_or_seed_or_budget_or_reward_or_oracle":True,
      "isolated_python_process_module_shadow_only":True,
      "candidate_runner_sha256":h(R/"tools/kci_kc8_real4b_train_diagnostic.py"),
      "candidate_scoring_module_sha256":h(R/"tools/kci_pisas_model_side_bridge.py"),
    }
    path.write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("PRECISION_COMPATIBILITY_PATCH_FROZEN",json.dumps({"HF_SHA":info["audited_HF_policy_sha256"],"original_protocol_sha":info["source_frozen_protocol_sha256"],"patch_SHA":h(path)},ensure_ascii=False))
if __name__=="__main__":main()
