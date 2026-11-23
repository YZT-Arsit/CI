"""Freeze ONLY existing KC8 TRAIN interface and native implementation sources.

Copies read-only Python source required by the KC8 env into a new isolated
algorithm study. Data AI worktree untouched. DEV, locked eval and native
Qstar artifacts are NOT copied. Native source is needed only in verifier,
never passed to the actor-visible model.
"""
from pathlib import Path
from hashlib import sha256
import json,tarfile
from datetime import datetime,timezone
BASE=Path(r"D:\research\kci_ci_causal_coverage_v2_20261122")
ROOT=Path(r"D:\research\longhistory")
OUT=ROOT/"artifacts/20261123_kc8_4b_policy_signal_v1"
SNAP=OUT/"KC8_NATIVE_CODE_SNAPSHOT_TRAIN_ONLY.tar.gz"
H=BASE/"artifacts/20261122_kci_causal_credit_method_handoff_v1"
FILES=(
 "__init__.py",
 "diverse_ci_branch_env.py",
 "kci_causal_task_contracts.py",
 "kci_causal_execution_env.py",
 "kci_causal_visible_surface.py",
 "kci_empirical_policy_env.py",
 "kci_empirical_candidate_roots.py",
 "kci_empirical_surface.py",
 "kci_policy_horizon_env.py",
 "kci_policy_horizon_surface.py",
 "kci_policy_task_contracts.py",
)
TRAIN=("MODEL_ONLY_TRAIN_A.jsonl","MODEL_ONLY_TRAIN_DOWNSTREAM.jsonl",
 "TRAIN_EARLY_A_ACTION_CE_IDENTICAL_STATES.jsonl",
 "TRAIN_ONE_PER_INDEPENDENT_STRICT_A_DPO.jsonl",
 "TRAIN_A_UNRANKED_TIE_NOT_DPO.jsonl",
 "TRAIN_STRICT_SIGNED_FEEDBACK_ACTION_CE.jsonl")
def sha(p):
 h=sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError("never overwrite existing study")
 src=BASE/"code/kci"
 assert len(FILES)==11 and len(TRAIN)==6
 for name in FILES:assert (src/name).is_file()
 for name in TRAIN:assert (H/name).is_file()
 hand=json.loads((H/"MANIFEST.json").read_text(encoding="utf8"))
 assert hand["independent_train_roots"]==6 and hand["independent_locked_test_roots_NOT_Qstar_solved"]==2
 outfile=OUT/"FROZEN_KC8_TRAIN_INPUTS"
 outfile.mkdir(parents=True)
 info={}
 for name in TRAIN:
  p=H/name
  dest=outfile/name
  dest.write_bytes(p.read_bytes())
  info["TRAIN_"+name]={"SHA256":sha(p),"bytes":p.stat().st_size}
 with tarfile.open(SNAP,"w:gz") as tar:
  for name in FILES:
   p=src/name
   tar.add(p,arcname="code/kci/"+name,recursive=False)
   info["CODE_"+name]={"SHA256":sha(p),"bytes":p.stat().st_size}
 for extra in ("verify_causal_credit_model_onpolicy_rollouts.py",):
  p=BASE/"tools"/extra
  assert p.is_file()
  with tarfile.open(SNAP,"a") if False else open(p,"rb") as f:pass
 # Standalone verifier copied to isolated study but actor never calls its native truth for choosing action.
 verifier=BASE/"tools/verify_causal_credit_model_onpolicy_rollouts.py"
 (OUT/"VERIFIER_ORIGINAL_SNAPSHOT.py").write_bytes(verifier.read_bytes())
 info["VERIFIER"]={"SHA256":sha(verifier),"bytes":verifier.stat().st_size}
 model_base=Path(r"D:\research\longhistory\artifacts\20261124_cce_downstream_sft_v1\shared_native_sft_seed7\adapter\adapter_model.safetensors")
 assert model_base.is_file()
 prereg={
  "status":"FROZEN_TRAIN_ONLY_SMALL_4B_QPI_KC8_DIAGNOSTIC",
  "purpose":"real trained shared 4B agent actions in new causal native train; no CCE/DPR optimization",
  "source_root":"D:\\research\\kci_ci_causal_coverage_v2_20261122",
  "source_branch":"data/kci-ci-causal-coverage-v2-20261122",
  "source_data_AI_unmodified":True,
  "n_independent_train_roots":6,
  "train_root_ids":["KC801","KC802","KC803","KC804","KC805","KC806"],
  "excluded_DEV":["KC807","KC808"],
  "excluded_LOCKED":["KC809","KC810"],
  "two_histories":["H1","H2"],
  "full_episode_seeds":[920031,920043],
  "full_episode_id":0,
  "n_full_model_episodes":24,
  "model_base":"Qwen3-4B-Instruct-2507-ms",
  "checkpoint_SHA256":sha(model_base),
  "model_adapter_origin":"artifacts/20261124_cce_downstream_sft_v1/shared_native_sft_seed7/adapter",
  "model_inference_quantization":"NF4/BF16 training-matched FP32 unquantized head",
  "one_shared_adapter_for_all_roles":True,
  "model_choice":"semantic_accept_reject per native action, one softmax categorical sample (temperature=1), no rule-based prefix except mandated two historical deliveries",
  "actor_observation":"EXACT data-AI kci_causal_visible_surface.model_only from current role-observation; no labels",
  "native_event_rng":"official deterministic_rng(sha256(seed|root|history|episode))",
  "chance_outcomes":"official choose_chance",
  "no_fabricated_next_stage":True,
  "full_episode_max_native_events":45,
  "max_prompt_tokens":8192,
  "no_truncation":True,
  "training_updates":0,
  "same_state_fork_selection":"hash-sorted root/history states from 12 prior-delivery prefixes before Qpi rollouts",
  "fork_n_distinct_train_roots":4,
  "fork_K_per_A_action":8,
  "fork_seeds":[930101,930107,930109,930113,930127,930131,930137,930149],
  "same_state_A_candidates":["SEND_A","SEND_B"],
  "K8_is_exploratory_NOT_confirmatory":True,
  "bonferroni_CI_first_threshold":0.05/(4*2),
  "min_paired_discordant":5,
  "unknown_blocks_label":True,
  "zero_observed_CI_violation_not_equivalent_to_zero_risk":True,
  "native_Qstar_NOT_a_teacher_for_Qpi":True,
  "terminal_reward_proxy":"safe_business_success+1 / observed_CI_violation-1 / others0",
  "DPR_proof_gate":"actual model failure+same_role_same_state legal alternative+real verified safe completion, otherwise unresolved/witness only",
  "paper_target_not_yet_supported":True,
  "all_input_checksums":info,
  "code_snapshot_sha256":sha(SNAP),
  "verifier_sha256":sha(OUT/"VERIFIER_ORIGINAL_SNAPSHOT.py"),
  "protocol_freeze_readonly":True,
 }
 (OUT/"FROZEN_PROTOCOL.json").write_text(json.dumps(prereg,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 print("KC8_TRAIN_FROZEN",json.dumps({"roots":6,"full_episodes":24,
   "source_files":len(info),"snapshot_sha":prereg["code_snapshot_sha256"],
   "adapter_sha":prereg["checkpoint_SHA256"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
