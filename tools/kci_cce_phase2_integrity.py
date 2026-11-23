"""Finalize CCE Phase2 mechanism artifacts with full SHA256 evidence manifest."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1"
MODEL=ROOT/"artifacts/20261123_shared_semantic_action_v1/train_shared_joint3_seed7/adapter/adapter_model.safetensors"
ORIGINAL=Path(r"D:\research\kci_data_v4_phase2_20261122")
FILES=(ROOT/"code/kci/v4_phase2_workflows.py",ORIGINAL/"code/kci/v4_phase2_workflows.py",
       ROOT/"code/kci/v4_phase2_data.py",ORIGINAL/"code/kci/v4_phase2_data.py")
REPORT=ROOT/"docs/KCI_PHASE2_CCE_MECHANISM_20261123_CN.md"
def hash(path):
    digest=sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            digest.update(chunk)
    return digest.hexdigest()
def j(path):return json.loads(path.read_text(encoding="utf8"))
def jl(path):return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
def main():
    assert REPORT.exists() and REPORT.stat().st_size>4000
    assert hash(FILES[0])==hash(FILES[1])
    assert hash(FILES[2])==hash(FILES[3])
    states=jl(D/"FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl")
    full=jl(D/"POLICY_SAMPLED_4B_K8/SAMPLED_ACTUAL_FROZEN_4B_TRAJECTORIES.jsonl")
    meta=j(D/"POLICY_SAMPLED_4B_K8/SAMPLE_SUMMARY.json")
    pre=j(D/"PRE_REGISTERED_CELL.json")
    contract=j(D/"CCE_COMPARISON_CONTRACT.json")
    cce=j(D/"CCE_ANALYSIS_K8/CCE_MECHANISM_SUMMARY.json")
    pairs=jl(D/"CCE_ANALYSIS_K8/CCE_PAIR_PREFERENCES.jsonl")
    stability=j(D/"TERMINAL_PROPAGATION_STABILITY.json")
    native=j(D/"CCE_ANALYSIS_K8/NATIVE_BOUND_QSTAR_AUDIT_ONLY.json")
    assert set(native)=={"P201","P202","P203","P205","P206","P207","P208"}
    assert len(states)==len(pairs)==14 and len(full)==224
    assert pre["output_sha256"]==hash(D/"FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl")
    assert contract["source_frozen_input_SHA256"]==pre["output_sha256"]
    assert contract["frozen_policy_adapter_SHA256"]==hash(MODEL)==meta["model_adapter_sha256"]
    assert tuple(contract["paired_sample_seeds"])==tuple(meta["sample_seeds"])
    assert meta["n_native_violations"]==cce["all_Qpi_branch_outcomes"]["CI_violations"]==11
    assert meta["n_safe_task_successes"]==cce["all_Qpi_branch_outcomes"]["CI_safe_success"]==53
    assert cce["CCE_accepted"]==0 and cce["CCE_abstained"]==14
    assert cce["terminal_accepted_no_uncertainty_gate"]==9
    assert stability["summary"]["n_jackknife_changed_including_tie"]==5
    assert all(x["root"] in native and x["condition"] in ("risky","authorized") for x in full)
    assert all(x["is_actual_frozen_4B_Qpi_sample_not_oracle_Qstar"] for x in full)
    assert all(x["A_immediate_verdict"]=="VALID" and not x["A_immediate_CI_violation"] for x in full)
    assert all(x["CCE_preference"] is None and x["CCE_abstain_reason"] for x in pairs)
    assert not any(x["unknown_or_unfinished"] for x in full)
    # Pilot and pytest outputs are explicitly NOT primary evidence.
    artifacts=sorted([p for p in D.rglob("*") if p.is_file() and p.name!="ARTIFACT_SHA256.json"
        and not any(s.startswith("pytest_") for s in p.relative_to(D).parts)],
        key=lambda p:p.as_posix())
    manifest={
      "status":"CCE_REAL_QPI_K8_MECHANISM_NEGATIVE_EVIDENCE_COMPLETE",
      "sampled_4B_native_team_trajectories":len(full),
      "n_true_same_state_A_pairs":len(states),
      "n_business_roots":7,"n_distinct_boolean_knowledge_mechanisms_after_equivalence":5,
      "K_per_A":8,
      "actual_future_CI_violations":11,"actual_CI_safe_task_completions":53,
      "CCE_pref_accepted":0,"CCE_abstained":14,
      "ordinary_terminal_given_directed_label":9,
      "terminal_labels_instability_in_k8_accepted":5,
      "model_adapter_sha256":hash(MODEL),
      "Phase2_workflow_source_SHA256":hash(FILES[0]),
      "Phase2_renderer_source_SHA256":hash(FILES[2]),
      "Phase2_source_code_verified_byte_identical":True,
      "original_Phase2_data_worktree_read_only":True,
      "never_used_DEV_or_final_test_or_DG36":True,
      "bounded_Qstar_not_labeled_as_frozen_model_Qpi":True,
      "not_a_CCE_policy_training_improvement_result":True,
      "report":"docs/KCI_PHASE2_CCE_MECHANISM_20261123_CN.md",
      "files":[{"relative_path":p.relative_to(D).as_posix(),
           "sha256":hash(p),"bytes":p.stat().st_size,
           "is_primary_study_evidence":not "PILOT_" in p.relative_to(D).as_posix()} for p in artifacts]
    }
    out=D/"ARTIFACT_SHA256.json"
    assert not out.exists()
    out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_ARTIFACT_INTEGRITY_OK",
        "evidence_files",len(artifacts),
        "model_hash",manifest["model_adapter_sha256"],
        "same_root",len(states),"Qpi_native_traces",len(full),
        "accepted",0,"abstained",14)
if __name__=="__main__":main()
