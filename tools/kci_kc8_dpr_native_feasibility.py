"""KC8 real on-policy failure DPR: same-state native alternative feasibility.

Original Qwen3-4B failures are independently replayed with original signed
tool feedback and real actor observations. Alternative actions are tested at
the exact failing state, same role, using the untouched official CausalEnv.
An evaluator-only exact finite native contingent solver is a REACHABILITY
UPPER BOUND, NEVER Qpi or a training teacher. No CCE/DPR labels exported.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from importlib import util
import json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_kc8_real4b_train_diagnostic import load_native_snapshot,clone_prefix,file_sha,D
FULL=D/"REAL4B_TRAIN_FROZEN_DIAGNOSTIC/REAL_4B_TRAIN_FULL_MODEL_NATIVE_EPISODES.jsonl"
SEL=D/"DPR_ACTUAL_FAILING_MODEL_STATES_PRE_ALT_FREEZE.json"
SRC=D/"EVAL_NATIVE_FINITE_SOLVER_READONLY_SNAPSHOT.py"
OUT=D/"DPR_REAL_MODEL_FAILURE_NATIVE_FEASIBILITY_EVAL_ONLY.json"
def sha(p):
 h=sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def jl(p):return [json.loads(x) for x in Path(p).open(encoding="utf8") if x.strip()]
def native_solver():
    import kci
    assert SRC.is_file() and SRC.stat().st_size>3000
    spec=util.spec_from_file_location("kci.kci_policy_exact_solver",SRC)
    module=util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module
def find_before_failure(row,select,env,verifier,model_only,digest):
    events=row["official_full_episode_record"]["events"]
    ix=select["failing_event_index"]
    assert 0<=ix<len(events)
    chance=verifier.deterministic_rng(row["root_id"],row["history"],row["seed"],row["episode_id"])
    st=clone_prefix(env)
    for j,event in enumerate(events[:ix]):
        assert event["before_native_state_sha256"]==digest(st.__dict__)
        if st.stage in ("PRIMARY_CHANCE","SECOND_CHANCE","AUTH_CHANCE"):
            wanted,after=verifier.choose_chance(env.chance_outcomes(st),chance)
            assert event["type"]=="chance" and event["outcome"]==wanted
            assert event["signed_feedback_signature"]==after.signed_feedback["signature"]
        else:
            obs=env.actor_observation(st);view=model_only(obs)
            assert event["type"]=="model_action"
            assert event["observed_prompt_sha256"]==view["prompt_sha256"]
            assert event["actor"]==obs["actor"]
            after,report=env.transition(st,event["action"])
            assert report["verdict"] in ("VALID","VIOLATION")
        assert event["after_native_state_sha256"]==digest(after.__dict__)
        st=after
        assert not st.done,(row["root_id"],j,"terminated before fail")
    expected=events[ix]
    assert expected["type"]=="model_action"
    assert digest(st.__dict__)==select["decision_before_sha256"]==expected["before_native_state_sha256"]
    assert expected["action"]==select["observed_model_action_or_signed_tool_event"]
    assert expected["actor"]==env.actor_observation(st)["actor"]
    assert not st.done
    original_after,original_report=env.transition(st,expected["action"])
    assert original_report["verdict"] in ("VALID","VIOLATION")
    assert digest(original_after.__dict__)==expected["after_native_state_sha256"]
    assert original_after.done and (original_after.violation or original_after.aborted)
    return st,original_after,original_report
def main():
    if OUT.exists():raise FileExistsError(OUT)
    cases_fn,CausalEnv,model_only,digest,verifier=load_native_snapshot()
    allcases={c.id:c for c in cases_fn() if c.split=="train" and c.id.startswith("KC8")}
    assert len(allcases)==6
    freeze=json.loads(SEL.read_text(encoding="utf8"))
    assert sha(FULL)==freeze["source_actual_24_model_episode_file_sha256"]
    assert len(freeze["selected_failure_states"])==4
    original=jl(FULL)
    by={(x["root_id"],x["history"],x["seed"]):x for x in original}
    solver=native_solver()
    assert "NOT_LLM_Qpi" in solver.__doc__ or "not sampled" in solver.__doc__.lower()
    rows=[];n_solve=0;start=time.monotonic()
    for sel in freeze["selected_failure_states"]:
        row=by[(sel["root"],sel["history"],sel["seed"])]
        env=CausalEnv(allcases[sel["root"]],sel["history"])
        before,after_orig,verdict_orig=find_before_failure(row,sel,env,verifier,model_only,digest)
        legal=list(env.options(before))
        actor=env.actor_observation(before)["actor"]
        assert actor==sel["actor"]
        assert sel["observed_model_action_or_signed_tool_event"] in legal
        original_know={env.target:sorted(env.knowledge(before,env.target))}
        candidate=[]
        for action in legal:
            if action==sel["observed_model_action_or_signed_tool_event"] or action=="ABORT":continue
            after,verdict=env.transition(before,action)
            if verdict["verdict"]!="VALID" or after.violation:
                candidate.append({"action":action,"native_immediate":verdict["verdict"],
                                  "native_safe_success_prob_upper_Qstar":None,
                                  "status":"REJECT_NOT_IMMEDIATE_CI_VALID"});continue
            fromtime=time.monotonic()
            exact=solver.solve(env,after)
            n_solve+=1
            value=exact["value"]
            assert len(value)==3
            prob=str(value[0])
            assert Fraction(value[0])>=0
            grade=("NATIVE_CONTINGENT_SAFE_COMPLETION_HAS_FEASIBLE_PATHS"
                   if value[0]>0 else "NATIVE_NO_SAFE_COMPLETION_PROBABILITY")
            candidate.append({
              "action":action,"original_actor":actor,
              "native_immediate":verdict["verdict"],
              "after_action_state_SHA256":digest(after.__dict__),
              "new_recipient_knowledge":{env.target:sorted(env.knowledge(after,env.target))},
              "native_safe_success_prob_upper_Qstar":prob,
              "native_future_violation_prob_optimal":str(-value[1]),
              "native_expected_total_elapsed_Qstar":str(-value[2]),
              "native_finite_solver_stats":exact["stats"],
              "solver_seconds":round(time.monotonic()-fromtime,5),
              "status":grade,
              "certification_grade":"EXACT_WITHIN_FINITE_STOCHASTIC_NATIVE_NOT_CURRENT_MODEL_Qpi",
              "can_export_DPR_training_label":False})
        r={
         "root":sel["root"],"history":sel["history"],"seed":sel["seed"],
         "failure_original_stage":sel["stage"],"failure_role":actor,
         "failure_original_model_action":sel["observed_model_action_or_signed_tool_event"],
         "same_state_role_visible_observation_sha256":digest(env.actor_observation(before)),
         "actual_received_knowledge_at_failure":original_know,
         "exact_state_sha256":digest(before.__dict__),
         "original_native_failure_report":verdict_orig,
         "original_real_model_CI_violation":row["CI_violation"],
         "original_real_model_safe_abort":row["safe_abort"],
         "original_real_model_terminal_step":row["native_elapsed"],
         "same_state_other_actions":candidate,
         "n_native_exact_eval_branches":len(candidate),
         "has_native_feasible_alternative":any(x.get("native_safe_success_prob_upper_Qstar")
                 not in (None,"0") for x in candidate),
         "real_model_Qpi_after_alternative_NOT_RUN":True,
         "DPR_trainability":"ABSTAIN_NEED_FROZEN_MODEL_OR_VERIFIED_ACCEPTABLE_FUTURE_WITNESS",
         "Qstar_not_model_Qpi":True,
         "no_original_model_or_dataset_change":True}
        rows.append(r)
        print("DPR_NATIVE_ACTUAL_FAILURE_REPLAY",sel["root"],sel["stage"],
          sel["real_failure_kind"],"native_valid_potential",r["has_native_feasible_alternative"],
          "other",[(z["action"],z["native_safe_success_prob_upper_Qstar"]) for z in candidate],flush=True)
    assert len(rows)==4
    report={
      "status":"TRAIN_4B_FAILURE_LOCAL_NATIVE_FEASIBILITY_ONLY_NO_DPR_TRAINING",
      "source_real_4B_full_episodes_sha256":sha(FULL),
      "source_pre_alt_selection_sha256":sha(SEL),
      "source_data_AI_exact_finite_solver_sha256":sha(SRC),
      "original_stated_source_solver_not_modified":True,
      "n_original_model_real_failure_states":len(rows),
      "n_separate_train_business_roots":len({x["root"] for x in rows}),
      "n_irreversible_signed_chance_violations_without_local_model_decision":freeze["signed_auto_source_violations_have_NO_role_local_model_action_in_chance_event"],
      "n_full_native_verifier_and_prefix_replayed":len(rows),
      "n_separate_native_stochastic_expected_Qstar_evaluations":n_solve,
      "candidate_native_feasible_alternatives_count":sum(x["has_native_feasible_alternative"] for x in rows),
      "actual_Qpi_repair_suffices_or_safe_success_UNKNOWN":True,
      "no_DPR_train_preferences_produced":True,
      "not_sufficient_to_prove_model_credit_algorithm":True,
      "native_solver_seconds_all":round(time.monotonic()-start,3),
      "failures":rows,
    }
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("DPR_NATIVE_LOCAL_FEASIBILITY_EVIDENCE_FINISHED",len(rows),"actual4B failures",n_solve,
          "native evaluator calls","DPO_DPR_LABELS",0,flush=True)
if __name__=="__main__":main()
