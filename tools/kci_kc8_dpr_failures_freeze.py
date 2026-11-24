"""DPR actual failing model-state selection BEFORE alternative evaluation.

Native Qstar and model continuation NOT read to select failures. One actual
4B model-caused CI violation, then upto three role-local ABORTs from different
TRAIN roots, selected by frozen hash order. Chance-only signed violations are
not falsely attributed to a model action at the chance step.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json
R=Path(__file__).resolve().parents[1]
D=R/"artifacts/20261123_kc8_4b_policy_signal_v1"
FULL=D/"REAL4B_TRAIN_FROZEN_DIAGNOSTIC/REAL_4B_TRAIN_FULL_MODEL_NATIVE_EPISODES.jsonl"
OUT=D/"DPR_ACTUAL_FAILING_MODEL_STATES_PRE_ALT_FREEZE.json"
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 p=json.loads((D/"FROZEN_PROTOCOL.json").read_text(encoding="utf8"))
 rows=[json.loads(x) for x in FULL.open(encoding="utf8") if x.strip()]
 assert len(rows)==24 and set(x["root_id"] for x in rows)==set(p["train_root_ids"])
 classified=[];eligible=[]
 for row in rows:
    if row["task_safe_success"]:continue
    events=row["real_model_action_trace"]
    failing=[e for e in events if e["native_CI_violation_after"] or
             (e["event_type"]=="model_action" and e["native_action_or_chance"]=="ABORT")]
    assert failing and len(failing)==1, (row["root_id"],row["seed"],len(failing))
    event=failing[0]
    ismodel=(event["event_type"]=="model_action")
    typeval=("FIRST_NATIVE_CI_VIOLATION_FROM_CHANCE" if not ismodel
             else "MODEL_DIRECT_FIRST_CI_VIOLATION" if row["CI_violation"]
             else "MODEL_SAFE_TASK_ABORT")
    fact={"root":row["root_id"],"history":row["history"],"seed":row["seed"],
          "stage":event["native_stage"],"actor":event["model_role"],
          "real_failure_kind":typeval,
          "failing_event_index":event["event_index"],
          "decision_before_sha256":event["native_state_before_SHA256"],
          "observed_model_action_or_signed_tool_event":event["native_action_or_chance"],
          "full_original_episode_native_trace_sha256":sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
          "frozen_checkpoint_sha256":row["checkpoint_sha256"],
          "real_model_did_choose_event":ismodel,
          "actual_CI_violation":row["CI_violation"],
          "actual_safe_abort":row["safe_abort"]}
    classified.append(fact)
    if ismodel and event["native_stage"] not in ("A_SELECT",) and typeval in (
      "MODEL_DIRECT_FIRST_CI_VIOLATION","MODEL_SAFE_TASK_ABORT"):
         eligible.append(fact)
 assert sum(x["real_failure_kind"]=="FIRST_NATIVE_CI_VIOLATION_FROM_CHANCE" for x in classified)==3
 assert sum(x["real_failure_kind"]=="MODEL_DIRECT_FIRST_CI_VIOLATION" for x in classified)==1
 assert sum(x["real_failure_kind"]=="MODEL_SAFE_TASK_ABORT" for x in classified)==16
 violations=sorted([x for x in eligible if x["real_failure_kind"]=="MODEL_DIRECT_FIRST_CI_VIOLATION"],
    key=lambda r:sha256(f"FIXED_KC8_DPR|{r['root']}|{r['seed']}".encode()).hexdigest())
 assert len(violations)==1
 chosen=violations[:1]
 aborts=sorted([x for x in eligible if x["real_failure_kind"]=="MODEL_SAFE_TASK_ABORT" and x["root"]!=chosen[0]["root"]],
   key=lambda r:sha256(f"FIXED_KC8_DPR|{r['root']}|{r['history']}|{r['seed']}|{r['failing_event_index']}".encode()).hexdigest())
 used={x["root"] for x in chosen}
 for x in aborts:
    if x["root"] in used:continue
    chosen.append(x);used.add(x["root"])
    if len(chosen)==4:break
 assert len(chosen)==4 and len(used)==4
 freeze={
  "status":"FROZEN_REAL_MODEL_TRAIN_FAILURE_DPR_PRE_ALTERNATIVE_TEST",
  "frozen_model_input_checkpoint_sha256":p["checkpoint_SHA256"],
  "source_actual_24_model_episode_file_sha256":sha(FULL),
  "n_actual_full_model_episodes":len(rows),
  "n_original_model_or_auto_CI_violation_events":4,
  "signed_auto_source_violations_have_NO_role_local_model_action_in_chance_event":3,
  "same_state_model_disclosure_violation_count":1,
  "n_actual_safe_abort_events":16,
  "n_role_local_non_A_action_eligible":len(eligible),
  "selected_failure_states":chosen,
  "selection_rule":"one actor-caused CI violation + earliest hash-ranked safe ABORT from different TRAIN roots; max 4",
  "only_original_real_model_failures_used_to_select":True,
  "no_alt_native_solver_or_other_model_continuation_results_used_to_select":True,
  "failure_cases_without_any_alternative_must_stay_unresolved":True,
  "DPR_labels_not_authorized_until_safe_same_state_witness":True,
  "no_DEV_or_LOCKED_native_episodes":True,
  "original_dataAI_source_untouched":True,
 }
 OUT.write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 print("REAL_MODEL_DPR_FAILURE_STATES_FROZEN",
      len(chosen),"distinct_roots",[x["root"]+":"+x["stage"]+":"+x["real_failure_kind"] for x in chosen],
      "auto_signed_irreversible",3,flush=True)
if __name__=="__main__":main()
