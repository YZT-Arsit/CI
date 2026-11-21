r"""DG19 exact native causal intervention under ORIGINAL frozen SFT policy.

Compare 4 DIFFERENT, explicitly labeled policies:
  π_ref: original checkpoint frozen all later actions.
  π_fix_B: force only Coordinator's B OPEN/SKIP to native best (privileged).
  π_fix_release: force only source agent's RAW/PRIVATE to native best.
  π_fix_both: force both, leaving C policy unchanged.
All interventions take place at the SAME actual native role/state after A
and use real native Qstar queries ONLY for counterfactual causal diagnosis.
Never call these intervened values the frozen raw policy Qpi used to train.
"""
from pathlib import Path
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_exact_frozen_qpi_cpu import (
 ROWS,environment,prepare_A,ExactQpi,load_frozen_policy,qstar_A_from_original)
from solve_v31_longterm import native_best
OUT=ROOT/"artifacts"/"20261121_policy_gap_pilot_01"/"DG19_PRIVILEGED_CAUSAL_ROLE_INTERVENTION"
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir(parents=True)
target="DG19_M01__tradeoff__swap0"
row=ROWS[target]
baseline,meta=load_frozen_policy("action_sft",7)
def correct_role(env,state,obs,options,*,B=False,release=False,counter=None):
    phase=obs.get("current_stage")
    if (phase=="preflight" and B and "OPEN_PRIVATE_CHANNEL" in options
            and "SKIP_PRIVATE_CHANNEL" in options):
        allowed=("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")
    elif (phase=="release" and release and "SEND_RAW_R" in options
            and "VERIFY_PRIVATE_R" in options):
        allowed=("SEND_RAW_R","VERIFY_PRIVATE_R")
    else:return None
    scores={}
    for a in allowed:
        q=native_best(env.native,state.native,a)
        scores[a]=q["native_exact_Qstar"]
        if counter is not None:
            counter["native_oracle_states"]+=q["states_explored"]
            counter["native_oracle_search_calls"]+=1
    return max(scores,key=scores.get)
rows=[]
for tag,bfix,sfix in (
    ("unchanged_frozen_Qpi",False,False),
    ("privileged_correct_only_B",True,False),
    ("privileged_correct_only_source_RAW_VERIFY",False,True),
    ("privileged_correct_B_AND_source",True,True)):
    env=environment(row)
    source=prepare_A(env)
    count={"native_oracle_states":0,"native_oracle_search_calls":0}
    def callback(e,st,obs,actions):
        return correct_role(e,st,obs,actions,B=bfix,release=sfix,counter=count)
    model=ExactQpi(env,baseline,meta["checkpoint_sha256"],
      role_intervention=(callback if bfix or sfix else None))
    ev=model.evaluate_A_arms(source)
    rec={
        "policy_condition":tag,
        "role_B_oracle_intervention":bfix,
        "source_release_oracle_intervention":sfix,
        "Qpi_A_minus_B_only_original_when_no_intervention":ev["Qpi_A_minus_B"],
        "arms":ev["arms"],
        "native_prob_weighted_states":ev["cost"]["native_states"],
        "frozen_ref_checkpoint_sha256":meta["checkpoint_sha256"],
        "privileged_intervention_reference_cost":count,
        "extra_training_data_from_privileged_role_action":False,
    }
    rows.append(rec)
    print("DG19_INTERVENTION",tag,
          ev["Qpi_A_minus_B"],
          "A Q",ev["arms"]["PACKAGE_A"]["Qpi"],
          "B Q",ev["arms"]["PACKAGE_B"]["Qpi"],
          "oracle calls",count["native_oracle_search_calls"],flush=True)
native_star=qstar_A_from_original(row,verify=True)
assert abs(native_star["PACKAGE_A"]-native_star["PACKAGE_B"]-0.028)<1e-8
assert abs(rows[0]["Qpi_A_minus_B_only_original_when_no_intervention"]-(-0.048719277))<2e-6
report={
  "status":"DG19_NATIVE_VERIFIED_FROZEN_Qpi_PLUS_ROLE_SPECIFIC_INTERVENTIONS",
  "world":target,"source_graph":row["graph"],
  "Qstar_A_minus_B":0.028,"frozen_raw_Qpi_A_minus_B":rows[0]["Qpi_A_minus_B_only_original_when_no_intervention"],
  "data_and_model_parameters_never_changed":True,
  "important_interventions_are_privileged_oracle_not_Qpi_training_labels":True,
  "conditions":rows,
}
(OUT/"DG19_CAUSAL_INTERVENTIONS.json").write_text(
 json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
