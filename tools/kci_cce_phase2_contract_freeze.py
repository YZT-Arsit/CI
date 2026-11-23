"""Lock CCE Phase2 policy, comparison protocol and train-only roots before 4B sampling."""
from pathlib import Path
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools")]
from kci_cce_phase2_preference import (SEEDS,ALPHA,MIN_DISCORDANT,
    MIN_COST_COMPARABLE,TERMINAL_PREFERENCE_EPSILON)
OUT=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1"
ADAPTER=ROOT/"artifacts/20261123_shared_semantic_action_v1/train_shared_joint3_seed7/adapter/adapter_model.safetensors"
def sha(p):
    h=sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):h.update(c)
    return h.hexdigest()
def main():
    assert ADAPTER.is_file() and ADAPTER.stat().st_size>10_000_000
    ctx=json.loads((OUT/"PRE_REGISTERED_CELL.json").read_text(encoding="utf8"))
    assert ctx["n_distinct_current_A_states"]==14
    assert tuple(ctx["K_seeds_per_action"])==SEEDS
    contract={
        "root_task":"CCE mechanism real frozen current-shared-4B continuation",
        "frozen_policy_role":"single Qwen3-4B-Instruct-2507-ms pretrained base and one prior trained joint3 shared LoRA",
        "frozen_policy_base_server":"/root/autodl-tmp/kci_models/Qwen3-4B-Instruct-2507-ms",
        "frozen_policy_adapter_server":"/root/kci_native_4090_20261122/artifacts/20261123_shared_semantic_action_v1/train_shared_joint3_seed7/adapter",
        "frozen_policy_adapter_SHA256":sha(ADAPTER),
        "frozen_policy_not_trained_or_modified_in_cell":True,
        "A_current_legal_actions":["A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"],
        "A_prefix":"one signed native package read, single A_SEND state snapshot",
        "K_per_A":len(SEEDS),"paired_sample_seeds":list(SEEDS),
        "model_action_sampling":"categorical softmax over semantics scores; temperature 1.0",
        "continuation_policy":"same single frozen shared Qwen3-4B LoRA controls all downstream roles and tool choices",
        "CI_first_priority":["observed future CI violation rate",
                             "CI safe task success rate",
                             "native cost among matching safe successful trajectory pairs"],
        "paired_exact_binomial_test_one_sided_alpha":ALPHA,
        "minimum_discordant_pairs":MIN_DISCORDANT,
        "minimum_paired_safe_success_for_cost":MIN_COST_COMPARABLE,
        "if_CI_discordance_not_significant":"ABSTAIN without considering task success or cost",
        "if_task_discordance_not_significant":"ABSTAIN without considering cost",
        "terminal_scalar":{"CI_safe_success":1,"actual_CI_violation":-1,
                           "safe_incomplete_or_abort":0,
                           "simple_preference_min_abs_mean_gap":TERMINAL_PREFERENCE_EPSILON},
        "terminal_baseline_uses_same_sampled_trajectories":True,
        "cached_model_scores_without_changing_policy_actions":True,
        "uncertainty_assumption":"finite K paired sample only; no general Qpi proof",
        "evidence_stage":"signal_generation_only, no DPO/CE policy training this turn until evidence is shown",
        "bounded_native_Qstar_is_offline_independent_audit_only":True,
        "source_frozen_input_SHA256":ctx["output_sha256"],
        "root_selection_pre_registered":ctx["root_ids"],
        "variants_pre_registered":ctx["context_conditions"],
        "no_dev_or_final_test_root_use":True,
    }
    dest=OUT/"CCE_COMPARISON_CONTRACT.json"
    assert not dest.exists()
    dest.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_REFERENCE_POLICY_AND_COMPARE_PROTOCOL_FROZEN",json.dumps({
        "model_SHA256":contract["frozen_policy_adapter_SHA256"],
        "root_count":len(contract["root_selection_pre_registered"]),
        "A_states":14,
        "K_per_action":contract["K_per_A"],
        "future_native_4B_rollouts_planned":14*2*len(SEEDS),
        "no_training":True},ensure_ascii=False))
if __name__=="__main__":main()
