"""Freeze V4 semantic 4B three-arm truth, pair metrics and actual native traces.

Refuse to certify the final artifact if a checkpoint or 90 native episodes
are absent, or if DG36 appears in any evaluated root. No result rewriting.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/20261123_v4_semantic_control_v1"
METHODS=("aware_ce","aware_dpo","blind_ce")
def sha(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf8"))
def jl(path):return [json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s.strip()]
def main():
    data=read(OUT/"DATA_MANIFEST.json")
    assert data["records_by_training_arm"]==9 and data["dev_worlds"]==10
    assert data["no_DG36_deserialized_or_scored"]
    aware=jl(OUT/"TRAIN_HISTORY_AWARE.jsonl")
    blind=jl(OUT/"TRAIN_HISTORY_BLIND.jsonl")
    assert len(aware)==len(blind)==9
    for a,b in zip(aware,blind):
        assert a["prompt"]==b["prompt"]
        assert a["semantic_candidate_sha256"]==b["semantic_candidate_sha256"]
        assert b["chosen_action"]=="SEND_RAW_R"
    meta={}
    for method in METHODS:
        sub=OUT/f"train_{method}_seed7"
        j=read(sub/"TRAINING.json")
        assert j["method"]==method and j["steps"]==96
        assert j["seed"]==7 and j["lora_rank"]==8 and j["lora_alpha"]==16
        assert j["n_independent_train_states"]==9
        assert j["n_original_V4_option_records"]==18
        assert j["n_generated_extra_option_swaps"]==0
        assert j["one_shared_adapter_no_role_routing"]
        assert j["native_Qstar_not_LLM_Qpi"]
        p=sub/"adapter/adapter_model.safetensors"
        assert p.is_file() and p.stat().st_size>10000000
        meta[method]={"checkpoint_sha256":sha(p),"steps":j["steps"],"wall_train_seconds":j["wall_train_seconds"],
          "training_state_counts":j["teacher_label_distribution"],
          "training_data_sha256":j["train_input_sha256"],
          "candidate_forward_calls":j["forward_counts"]["train_semantic_candidate_forwards"],
          "frozen_reference_candidate_forward_calls":j["DPO_reference_candidate_forwards"],
          "sampling_schedule_sha256":j["sampling_schedule_sha256"]}
    assert len(set(v["sampling_schedule_sha256"] for v in meta.values()))==1
    assert meta["aware_ce"]["training_data_sha256"]==meta["aware_dpo"]["training_data_sha256"]
    assert meta["blind_ce"]["training_data_sha256"]!=meta["aware_ce"]["training_data_sha256"]
    score=read(OUT/"eval_5arms_semantic/SUMMARY.json")
    decisions=jl(OUT/"eval_5arms_semantic/DECISIONS.jsonl")
    assert len(decisions)==19 and score["n_dev_states"]==10 and score["n_native_train_states"]==9
    for r in decisions:
        assert r["root"] in ("DG31","DG32","DG33","DG34","DG35")
        assert r["position_order_invariance_exact"]
        assert len(r["views"])==(1 if r["split"]=="train" else 2)
    native=read(OUT/"native_agent_3arms_v2/SUMMARY.json")
    episodes=jl(OUT/"native_agent_3arms_v2/FULL_NATIVE_TEAM_EPISODES.jsonl")
    assert len(episodes)==native["full_native_episode_count"]==90
    assert {x["root"] for x in episodes}=={"DG34","DG35"}
    assert all(x["all_actions_attempted_against_native_env"] for x in episodes)
    assert all(x["split"]=="dev" for x in episodes)
    assert all("DG36" not in x["world"] for x in episodes)
    assert sum(x["protocol"]=="controlled_release" for x in episodes)==60
    assert sum(x["protocol"]=="autonomous_shared" for x in episodes)==30
    assert all(len(x["stepwise_trace"])>0 for x in episodes)
    report={
      "research_gate":"TRAIN_FLIP_NOT_LEARNED_SO_NO_COMPLEX_RL",
      "source_v4_read_only":data["source_V4_is_read_only"],
      "DG36_never_parsed_or_scored":True,
      "same_base_and_lora_configuration":True,
      "n_train_independent_roots":3,"n_train_independent_release_comparisons":9,
      "n_dev_independent_roots":2,"n_dev_native_worlds":10,
      "views":["native","entity_alias"],
      "position_swap_semantically_identical_by_prompt_byte_equivalence":True,
      "models":meta,
      "dev_result":{},
      "native_team":native["summary"],
      "native_team_episode_count":len(episodes),
      "native_attempted_model_actions":sum(x["model_decision_count"] for x in episodes),
      "native_incomplete_or_unknown_episodes":sum(x["native_incomplete_due_rejected_or_UNKNOWN_action"] for x in episodes),
      "native_CI_violations":sum(x["native_actual_violation"] for x in episodes),
      "native_successes":sum(x["native_actual_success"] for x in episodes),
      "training_is_supervision_not_current_policy_Qpi":True,
      "full_agent_autonomous_worlds_not_pairwise_counterfactual":True,
      "native_fixed_controlled_team_not_full_policy_Qpi_for_untrained_roles":True,
      "no_CCE_or_DPR_experiment_performed":True,
      "one_seed_no_general_learning_conclusion":True
    }
    for view in ("native","entity_alias"):
        report["dev_result"][view]={}
        for method in METHODS:
            s=score["models"]["dev"][view][method]
            report["dev_result"][view][method]={
              "states_correct":s["state_actions_correct"],
              "flip_correct":s["pair_metrics"]["Flip"]["both_correct"],
              "flip_den":s["pair_metrics"]["Flip"]["n_roots"],
              "hold_correct":s["pair_metrics"]["Hold"]["both_correct"],
              "hold_den":s["pair_metrics"]["Hold"]["n_roots"],
              "authorization_correct":s["pair_metrics"]["Authorized"]["both_correct"],
              "authorization_den":s["pair_metrics"]["Authorized"]["n_roots"],
              "safe_RAW":s["raw_on_legitimate"],"safe_RAW_den":s["legitimate_n"],
              "risky_RAW_violations":s["raw_on_risky"],"risky_den":s["risky_n"],
              "native_safe_success_fixed_suffix":s["native_fixed_suffix_safe_success"],
              "extra_native_steps_on_completed":s["native_fixed_suffix_extra_steps_sum"],
              "mean_Praw_risky":s["mean_P_RAW_risky"],"mean_Praw_legitimate":s["mean_P_RAW_legitimate"]}
    path=OUT/"RESEARCH_GATE_AND_EVIDENCE.json"
    assert not path.exists()
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    # Never advertise a model checkpoint if not present; hashes are
    # streaming SHA256 over complete copied weight bytes.
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="ARTIFACT_SHA256.json")
    hashes=[{"relative_path":p.relative_to(OUT).as_posix(),"sha256":sha(p),
             "size":p.stat().st_size} for p in files]
    manifest={"status":"V4_SEMANTIC_THREE_ARM_EXPERIMENT_COMPLETE",
        "n_files":len(hashes),"n_new_full_model_LoRA_checkpoints":3,
        "all_source_root_splits_preserved":True,
        "DG36_never_evaluated":True,
        "research_gate_result":"FAIL_0_3_TRAIN_FLIP",
        "report":"docs/KCI_V4_SEMANTIC_CONTROL_20261123_CN.md",
        "entries":hashes}
    output=OUT/"ARTIFACT_SHA256.json"
    assert not output.exists()
    output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("FINAL_V4_SEMANTIC_AUDIT_PASS",
      "file_count",len(files),"new_lora",3,
      "native_episodes",len(episodes),
      "native_invalid",report["native_incomplete_or_unknown_episodes"],
      "dev_flip",{m:report["dev_result"]["native"][m]["flip_correct"] for m in METHODS})
if __name__=="__main__":main()
