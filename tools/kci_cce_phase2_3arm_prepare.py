"""Freeze same-state A supervision from three different evidence sources.

  Action CE: single verified native safe-completion witness with shortest
             observed COST (not bounded Qstar, not current-model Qpi)
  Ordinary terminal propagation: unchanged +1/-1/0 mean on K8 frozen Qpi
  CCE: confirmatory CI-first paired K16/K32 preference only.
All three use exactly the intersection of CCE-accepted TRAIN A states,
same actor-visible observation and same candidate actions, never final.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
SFT=ROOT/"artifacts/20261124_cce_downstream_sft_v1"
CCE=ROOT/"artifacts/20261124_cce_phase2_post_sft_v1"
A=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1/FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl"
OUT=ROOT/"artifacts/20261124_cce_method_comparison_v1"
ACTIONS=("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")
def jl(p):return [json.loads(s) for s in Path(p).open(encoding="utf8") if s.strip()]
def sha(p):
    h=sha256()
    with Path(p).open("rb") as f:
        for ch in iter(lambda:f.read(1048576),b""):h.update(ch)
    return h.hexdigest()
def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    cce=jl(CCE/"POST_SFT_FROZEN_ADAPTIVE/ADAPTIVE_CELL_DECISIONS.jsonl")
    source=jl(A)
    witnesses=jl(SFT/"NATIVE_SAFE_FULL_TEAM_WITNESSES_TRAIN_ONLY.jsonl")
    assert len(cce)==len(source)==14
    index={x["cell_id"]:x for x in source}
    assert len(index)==14
    by_root_and_action=defaultdict(list)
    for w in witnesses:
        assert w["terminal_safe_task_success"]
        if w["variant"]=="risky" and not w["authorized"]:
            by_root_and_action[(w["root"],w["A_action"])].append(w["native_cost"])
    rows=[]
    for result in cce:
        pref=result["confirmed_CCE_preference"]
        if pref is None:continue
        key=result["cell_id"];s=index[key]
        assert result["root"]==s["root"] and s["condition"]=="risky"
        assert pref in ACTIONS
        final=result["look_history"][-1]["evidence"]["current_look"]
        assert final["confirmed"] and final["dimension"]=="CI_safety"
        assert final["decision"]==pref
        assert final["paired_sign"]["exact_two_sided_p"]<0.0017857143
        assert final["block_consistency"]["passed"]
        term=result["K8_terminal_scalar_propagation"]["chosen"]
        term_final=result["final_terminal_scalar_propagation"]["chosen"]
        assert term in ACTIONS and term_final in ACTIONS
        witness_cost={a:min(by_root_and_action[(s["root"],a)])
               if by_root_and_action[(s["root"],a)] else None for a in ACTIONS}
        valid=[(a,v) for a,v in witness_cost.items() if v is not None]
        assert valid
        m=min(v for a,v in valid)
        top=[a for a,v in valid if v==m]
        if len(top)>1:
            # Keep the ambiguity; do not invent CE label.
            action_ce_label=None
        else:action_ce_label=top[0]
        rows.append({
           "cell_id":key,"root":s["root"],"condition":s["condition"],
           "split":"train","actor":s["A_model_observation"]["actor"],
           "stage":s["A_model_observation"]["stage"],
           "role_observation":s["A_model_observation"],
           "role_visible_A_observation_sha256":s["A_pre_state_sha256"],
           "allowed_native_A_actions":list(s["A_legal_actions"]),
           "both_current_A_deliveries_are_native_VALID":s["current_A_both_actions_native_CI_legal"],
           "training_same_initial_A_state":True,
           "teacher_actions":{
               "action_ce_witness":action_ce_label,
               "terminal_K8":term,
               "terminal_adaptive_finalK":term_final,
               "CCE_confirmed":pref},
           "native_verified_full_success_witness_costs_by_A":witness_cost,
           "native_CE_teacher_note":"min actual fully validated safe-success witness cost; not exact Qstar; not policy Qpi",
           "terminal_teacher_note":"ordinary +1/-1/0 value propagated from same frozen 4B Qpi samples, without confidence",
           "CCE_evidence_paired_two_sided_p":final["paired_sign"]["exact_two_sided_p"],
           "CCE_evidence_n_discordant":final["paired_sign"]["n_discordant"],
           "CCE_evidence_look_K":result["K_final_per_action"],
           "CCE_accepted_from_frozen_actual_4B_Qpi_not_native_Qstar":True,
           "no_hidden_future_label_in_model_prompt":True})
    assert len(rows)==7 and len({r["root"] for r in rows})==7
    assert all(x["teacher_actions"]["action_ce_witness"] in ACTIONS for x in rows)
    assert all("ABORT" in x["allowed_native_A_actions"] for x in rows)
    assert all(x["role_observation"]["stage"]=="A_SEND" and x["role_observation"]["actor"]=="RecordsAgent" for x in rows)
    count=Counter()
    for x in rows:
        y=x["teacher_actions"]
        count.update({"terminal_equals_CCE":y["terminal_K8"]==y["CCE_confirmed"],
                      "ActionCE_equals_CCE":y["action_ce_witness"]==y["CCE_confirmed"]})
    OUT.mkdir(parents=True)
    p=OUT/"MATCHED_SAME_A_STATE_THREE_TEACHERS_TRAIN_ONLY.jsonl"
    p.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows),encoding="utf8")
    manifest={
      "status":"FROZEN_MATCHED_THREE_ARM_A_SUPERVISION",
      "teacher_types":["action_ce_witness","terminal_K8","CCE_confirmed"],
      "all_data_source_TRAIN_only":True,
      "independent_A_states":7,
      "independent_business_roots":7,
      "all_A_states_same_across_methods":True,
      "model_actor_visible_observation_identical_across_methods":True,
      "model_initialization_single_pretrained_Qwen3_4B_plus_frozen_shared_SFT_LoRA":True,
      "previous_SFT_adapter_hash":sha(SFT/"shared_native_sft_seed7/adapter/adapter_model.safetensors"),
      "base_model":"Qwen3-4B-Instruct-2507-ms",
      "three_method_supervision_labels_equal_counts":dict(count),
      "same_optimizer_for_all":True,
      "optimizer":"action_CE_softmax_over_native_semantic_candidate_actions",
      "frozen_training_steps_per_arm":48,
      "batch":1,"learning_rate":2e-5,"seed":7,
      "LoRA_r":8,"LoRA_alpha":16,
      "source_labels_SHA256":sha(p),
      "source_CCE_adaptive_preferences_SHA256":sha(CCE/"POST_SFT_FROZEN_ADAPTIVE/ADAPTIVE_CELL_DECISIONS.jsonl"),
      "source_original_safe_witnesses_SHA256":sha(SFT/"NATIVE_SAFE_FULL_TEAM_WITNESSES_TRAIN_ONLY.jsonl"),
      "no_DEV_or_final_test_evaluated_or_train_labels":True,
      "no_DPR_or_online_RL":True,
      "native_verified_witness_CE_is_not_Qpi_and_not_Qstar":True}
    (OUT/"THREE_ARM_PRE_REGISTRATION.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_A_THREE_TEACHERS_PREP",json.dumps(manifest,ensure_ascii=False,indent=2))
    for r in rows:print("MATCHED_A_TEACHERS",r["root"],r["teacher_actions"],r["native_verified_full_success_witness_costs_by_A"])
if __name__=="__main__":prepare()
