"""Independent read-only A/B experiment completion and provenance audit.

No model training, no GPU, original artifacts untouched. Crosschecks real
initial trajectories, role stages, all K32 fork rows and the original frozen
analysis; reports comparator cost, competence gates, absence of CCE signal.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
import json
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"artifacts/20261123_executor_coverage_completed_audit_v1"
B=ROOT/"artifacts/20261123_cce_reference_completion_v1"
OLD=ROOT/"artifacts/20261122_cce_discovery_repair_v2/final_snapshot/study"
def sha(p):
 h=sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def j(p):return json.loads(p.read_text(encoding="utf8"))
def jl(p):return [json.loads(x) for x in p.open(encoding="utf8") if x.strip()]
def audit_a():
 out=A/"AUDIT_COMPLETED_EXECUTOR.json"
 if out.exists():raise FileExistsError(out)
 dec=j(A/"DECISION.json")
 base=j(A/"baseline_EVAL.json");cand=j(A/"candidate_EVAL.json")
 tr=j(A/"TRAINING.json")
 assert base["model"]=="baseline" and cand["model"]=="candidate"
 assert dec["decision"]=="CAPABILITY_GATE_FAILED_NO_CCE_TRAINING"
 assert dec["gates"]["legal_RAW_preserved"]==False
 assert tr["adapter_sha256"]==cand["model_sha256"]
 assert base["model_sha256"]=="b4660a4991a08fa2e1009b1c2dfb25e73f3f311f290e878352bca6e4deaa567f"
 assert cand["model_sha256"]=="0d4d08a495b9f4204196b6a1161a46813793d9a9aaafa0e4fd5a23b302f32ecf"
 assert base["protocol_sha256"]==cand["protocol_sha256"]==tr["protocol_sha256"]
 assert base["panel_sha256"]==cand["panel_sha256"]
 assert tr["candidate_forwards"]==772
 assert len(tr["curve"])==160
 assert len(base["initial_episodes"])==len(cand["initial_episodes"])==14
 assert len(base["calibration_cases"])==len(cand["calibration_cases"])==73
 assert base["legal_RAW_retained"]==5 and cand["legal_RAW_retained"]==3
 assert base["initial_tasks"]["safe_success"]==10 and cand["initial_tasks"]["safe_success"]==11
 assert base["initial_tasks"]["violations"]==4 and cand["initial_tasks"]["violations"]==3
 assert base["calibration"]["safe_success"]==65 and cand["calibration"]["safe_success"]==70
 assert base["calibration"]["violations"]==8 and cand["calibration"]["violations"]==2
 assert {p["root"] for p in base["initial_episodes"]}=={p["root"] for p in cand["initial_episodes"]}
 result={
   "status":"PASS_READONLY_EXISTING_GPU_ONE_CANDIDATE_ALGORITHM_GATE_NOT_PROMOTED",
   "candidate_decision":dec["decision"],
   "all_gates":dec["gates"],
   "base_adaptor_sha256":base["model_sha256"],
   "candidate_adapter_sha256":cand["model_sha256"],
   "train_updates":160,"training_candidate_forwards":772,
   "training_wall_seconds":tr["wall_seconds_excluding_load"],
   "baseline_inference_forwards":base["candidate_forwards"],
   "candidate_inference_forwards":cand["candidate_forwards"],
   "baseline_tokens":base["input_tokens"],"candidate_tokens":cand["input_tokens"],
   "matched_native_panel_hash":base["panel_sha256"],
   "matched_frozen_protocol_hash":base["protocol_sha256"],
   "calibration_count":73,
   "initial_complete_task_count":14,
   "initial_task_effect":{
     "baseline_STSR":base["initial_tasks"]["safe_success"],
     "candidate_STSR":cand["initial_tasks"]["safe_success"],
     "baseline_CI_violations":base["initial_tasks"]["violations"],
     "candidate_CI_violations":cand["initial_tasks"]["violations"]},
   "legal_RAW":{"base":base["legal_RAW_retained"],"candidate":cand["legal_RAW_retained"],"den":7},
   "stage_comparison":{x:{"baseline":base["per_stage"][x],"candidate":cand["per_stage"][x]}
        for x in sorted(base["per_stage"])},
   "invalid_to_promote_due_overprotection":True,
   "no_retraining_this_turn":True,
   "original_native_env_and_original_logs_untouched":True,
   "gpu_study_original_location":"/root/kci_executor_coverage_20261122_v2/study",
   "files_sha256":{x.name:sha(x) for x in A.iterdir() if x.is_file()},
 }
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 print("EXECUTOR_FINISHED_READONLY_AUDIT_PASS","STSR 10/14->11/14","legalRAW 5/7->3/7",
   "candidate_sha256",result["candidate_adapter_sha256"],flush=True)
def audit_b():
 out=B/"INDEPENDENT_FROZEN_REFERENCE_AUDIT.json"
 if out.exists():raise FileExistsError(out)
 protocol=j(B/"PROTOCOL.json")
 analysis=j(B/"ANALYSIS.json")
 pool=j(B/"STATE_POOL.json")
 ref=jl(B/"reference_FORKS.jsonl")
 collector=jl(OLD/"collector_FORKS.jsonl")
 assert sha(B/"PROTOCOL.json")==sha(OLD/"PROTOCOL.json")
 assert sha(B/"STATE_POOL.json")==sha(OLD/"STATE_POOL.json")
 assert len(pool["states"])==4 and len(ref)==len(collector)==512
 assert len(jl(B/"reference_COMPETENCE.jsonl"))==28
 assert analysis["selected_states"]==4 and analysis["confirmed_labels"]==0
 assert not analysis["stage_C_signal_gate_passed"]
 assert analysis["decision"]=="SIGNAL_NOT_ESTABLISHED_NO_C_TRAINING"
 assert analysis["competence"]["collector"]["safe_success"]==20 and analysis["competence"]["reference"]["safe_success"]==10
 assert analysis["competence"]["collector"]["passed"] and not analysis["competence"]["reference"]["passed"]
 assert len(j(B/"CONFIRMED_LABELS.json"))==0
 assert protocol["collector_sha256"]=="b4660a4991a08fa2e1009b1c2dfb25e73f3f311f290e878352bca6e4deaa567f"
 assert protocol["reference_sha256"]=="2e0052867d58d354fc5ca8059c0fe73255e140752cda6b13f823e319e14c737c"
 assert len(protocol["discovery_seeds"])==len(protocol["validation_seeds"])==32
 ids={r["decision_id"] for r in pool["states"]}
 for policy,rows in (("collector",collector),("reference",ref)):
  assert len(set((r["decision_id"],r["block"],r["seed"],r["action"]) for r in rows))==512
  assert set(r["decision_id"] for r in rows)==ids
  assert {r["policy_id"] for r in rows}=={protocol[policy+"_sha256"]}
  for r in rows:
   assert r["block"] in ("discovery","validation")
   assert r["seed"] in protocol[r["block"]+"_seeds"]
 for cell in analysis["cells"]:
  assert not cell["independently_confirmed_across_named_policies"]
  for policy in ("collector","reference"):
   for block in ("discovery","validation"):
    x=cell["estimates"][policy][block]
    assert x["cce_preference"] is None and x["status"].startswith("ABSTAIN")
 unique_dirs=Counter()
 statuses=Counter()
 for cell in analysis["cells"]:
  for policy in ("collector","reference"):
   for block in ("discovery","validation"):
    x=cell["estimates"][policy][block]
    unique_dirs[(policy,block,str(x["mean_return_preference"]))]+=1
    statuses[(policy,block,x["status"])]+=1
 result={
  "status":"PASS_FROZEN_REFERENCE_EXACT_CHECKPOINT_K32_NO_NEW_SIGNAL",
  "n_eligible_unique_states":pool["eligible_unique_states"],
  "n_all_visited_decisions":pool["visited_occurrences"],
  "n_exact_model_observation_states":pool["unique_visited_states"],
  "n_frozen_forks":len(pool["states"]),
  "n_real_4B_model_forks_total":len(ref)+len(collector),
  "reference_complete_forks":len(ref),
  "collector_complete_forks":len(collector),
  "frozen_state_pool_sha256":sha(B/"STATE_POOL.json"),
  "frozen_input_hashes":len(protocol["input_hashes"]),
  "collector_SHA256":protocol["collector_sha256"],
  "reference_SHA256":protocol["reference_sha256"],
  "n_train_roots_represented":len({x["root"] for x in pool["states"]}),
  "model_competence":analysis["competence"],
  "confirmed_training_preferences":analysis["confirmed_labels"],
  "new_exact_observation_states":analysis["new_exact_observation_states"],
  "analysis_decision":analysis["decision"],
  "abstain_status_counts":{str(k):v for k,v in statuses.items()},
  "ordinary_unfiltered_scalar_direction_counts":{str(k):v for k,v in unique_dirs.items()},
  "candidate_model_costs":{m:j(B/(m+"_COST.json")) for m in ("collector","reference")},
  "original_collector_forks_sha256":sha(OLD/"collector_FORKS.jsonl"),
  "new_reference_forks_sha256":sha(B/"reference_FORKS.jsonl"),
  "report_is_a_readonly_recheck_not_new_sampling":True,
  "old_reference_run_protocol_unchanged":True,
  "no_new_CCE_or_DPR_training_labels":True,
 }
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 print("FROZEN_REFERENCE_B_ROUNDTRIP_AUDIT_PASS","K32","512+512",len(ids),"states", "confirmed=0",flush=True)
if __name__=="__main__":
 audit_a();audit_b()
