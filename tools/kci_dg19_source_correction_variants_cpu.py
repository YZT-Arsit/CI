r"""Why previously reported '+0.01308 source-only' depends on intervention.

From SAME original source checkpoint and DG19 state, compare distinct,
fully specified, *privileged* source-agent remediation policies. This is
causal interpretation, NOT base-policy Qpi or training data.
"""
from pathlib import Path
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_exact_frozen_qpi_cpu import ROWS,environment,prepare_A,ExactQpi,load_frozen_policy
from solve_v31_longterm import native_best
OUT=ROOT/"artifacts"/"20261121_policy_gap_pilot_01"/"DG19_SOURCE_ONLY_INTERVENTION_DEFINITION_CHECK"
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir()
model,identity=load_frozen_policy("action_sft",7)
row=ROWS["DG19_M01__tradeoff__swap0"]
results=[]
for mode in (
    "source_RAW_VERIFY_only_if_both_legal",
    "source_OPTIMAL_among_RAW_VERIFY_ABORT_all_release_states",
    "source_SAFE_if_raw_violation_then_private_or_abort",
    "source_always_private_if_available_else_model",
):
    stats={"qstar_calls":0,"qstar_states":0,"overrides":0}
    def intervene(env,state,obs,actions):
        if obs.get("current_stage")!="release" or "SEND_RAW_R" not in actions:
            return None
        choices=None
        if mode=="source_RAW_VERIFY_only_if_both_legal":
            if "VERIFY_PRIVATE_R" in actions:
                choices=("SEND_RAW_R","VERIFY_PRIVATE_R")
        elif mode=="source_OPTIMAL_among_RAW_VERIFY_ABORT_all_release_states":
            choices=tuple(a for a in actions if a in
                          ("SEND_RAW_R","VERIFY_PRIVATE_R","ABORT"))
        elif mode=="source_SAFE_if_raw_violation_then_private_or_abort":
            n,a=env.transition(state,"SEND_RAW_R")
            if a["verdict"]=="VIOLATION":
                return "VERIFY_PRIVATE_R" if "VERIFY_PRIVATE_R" in actions else "ABORT"
            return "SEND_RAW_R"
        elif mode=="source_always_private_if_available_else_model":
            return "VERIFY_PRIVATE_R" if "VERIFY_PRIVATE_R" in actions else None
        if choices:
            q={}
            for a in choices:
                obj=native_best(env.native,state.native,a)
                q[a]=obj["native_exact_Qstar"]
                stats["qstar_calls"]+=1
                stats["qstar_states"]+=obj["states_explored"]
            stats["overrides"]+=1
            return max(q,key=q.get)
        return None
    env=environment(row)
    v=ExactQpi(env,model,identity["checkpoint_sha256"],
               role_intervention=intervene).evaluate_A_arms(prepare_A(env))
    results.append({
      "mode":mode,
      "Qpi_A_minus_B_under_this_privileged_intervention":v["Qpi_A_minus_B"],
      "A_expected_return":v["arms"]["PACKAGE_A"]["Qpi"],
      "B_expected_return":v["arms"]["PACKAGE_B"]["Qpi"],
      "intervention_search_cost":stats,
      "not_original_unmodified_Qpi":True,
    })
    print("SOURCE_DEF",mode,v["Qpi_A_minus_B"],v["arms"]["PACKAGE_A"]["Qpi"],
          v["arms"]["PACKAGE_B"]["Qpi"],flush=True)
report={
 "status":"DG19_SOURCE_ONLY_ROLE_INTERVENTIONS_ARE_OPERATIONAL_DEFINITION_DEPENDENT",
 "source_ref_checkpoint_sha256":identity["checkpoint_sha256"],
 "exact_unmodified_Qpi_AminusB":-0.048719277,
 "user_supplied_source_only_reference_number":0.01308,
 "results":results,
 "raw_Qpi_value_for_policy_training_always_unchanged":True,
}
(OUT/"SOURCE_INTERVENTIONS.json").write_text(json.dumps(
  report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
