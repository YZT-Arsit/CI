"""TRAIN-only local proof-closure shortcut audit, NO future rollout/Qstar.

Uses exactly current A ROLE-LEGAL observation: signed package contents,
public rules, required future source facts, explicit current authorization.
The heuristic is intentionally cheap and NOT an oracle policy: assume RAW
delivery of required facts, ask if a protected association becomes derivable.
Compare to already frozen post-SFT CCE signals only after deriving each
prediction without looking at their labels.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261124_cce_phase2_post_sft_v1"
PRE=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1/FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl"
SOURCE=D/"POST_SFT_FROZEN_ADAPTIVE/ADAPTIVE_CELL_DECISIONS.jsonl"
def jl(p):return [json.loads(l) for l in p.open(encoding="utf8") if l.strip()]
def derivable(known,rules):
    result=set(known)
    for _ in range(20):
        new=result|{r["entails"] for r in rules if set(r["requires"])<=result}
        if len(new)==len(result):return result
        result=new
    raise RuntimeError("Public rule closure did not converge")
def compute(row):
    ob=row["A_model_observation"]
    assert ob["actor"]=="RecordsAgent" and ob["stage"]=="A_SEND"
    packages=ob["source_bundle_tool_output"]
    assert set(packages)=={"PACKAGE_A","PACKAGE_B"}
    rules=ob["public_inference_rules"]
    required=set(ob["required_evidence"])
    # Actual A recipient norm is role-legally visible. This is a static
    # counterfactual *risk* check, NOT a Qpi roll-out or claim of impossibility.
    norm=ob["privacy_policy"]
    authorized="A purpose-scoped authorization explicitly permits" in norm
    assert authorized==(row["condition"]=="authorized")
    result={}
    for pack in ("PACKAGE_A","PACKAGE_B"):
        signed=packages[pack]
        assert signed["signed"] and signed["recipient"]==ob["recipient"]
        # Phase2 publicly maps facts a-f to signed source slots S01-S06.
        native_facts={"S%02d"%(ord(f)-ord("a")+1) for f in signed["facts"]}
        prior={ev["fact"] for ev in ob["delivered_message_log"] if ev["recipient"]==ob["recipient"]}
        prior_slots={"S%02d"%(ord(f)-ord("a")+1) for f in prior if len(f)==1 and "a"<=f<="f"}
        now=derivable(prior_slots|native_facts,rules)
        full=derivable(prior_slots|native_facts|required,rules)
        risk_after_more_sources= "protected association" in full and "protected association" not in now and not authorized
        result["A_SEND_"+pack]={
           "signed_native_package_fact_slots":sorted(native_facts),
           "knowledge_current_closure":sorted(now),
           "closure_if_later_all_task_required_sources_are_raw":sorted(full),
           "projected_unlicensed_protected_inference":risk_after_more_sources,
           "static_without_true_4B_policy":True}
    hazard={a:v["projected_unlicensed_protected_inference"] for a,v in result.items()}
    if hazard["A_SEND_PACKAGE_A"]!=hazard["A_SEND_PACKAGE_B"]:
        chosen=next(a for a in hazard if not hazard[a])
    else:chosen=None
    return {"cell_id":row["cell_id"],"root":row["root"],
       "condition":row["condition"],"static_local_heuristic_chosen":chosen,
       "package_audit":result,"both_A_current_native_CI_legal":row["current_A_both_actions_native_CI_legal"],
       "no_future_Qpi_or_Qstar_in_heuristic_input":True}
def main():
    if (D/"STATIC_LOCAL_CLOSURE_AUDIT.json").exists():raise FileExistsError("Do not overwrite")
    freeze=jl(PRE)
    assert len(freeze)==14
    predicted=[compute(x) for x in freeze]
    cce=jl(SOURCE)
    assert len(cce)==14
    lookup={x["cell_id"]:x for x in cce}
    for row in predicted:
        q=lookup[row["cell_id"]]
        row["CCE_accepted_action_joined_AFTER_heuristic_computation"]=q["confirmed_CCE_preference"]
        row["matches_confirmed_CCE_when_available"]=(
          None if q["confirmed_CCE_preference"] is None else
          row["static_local_heuristic_chosen"]==q["confirmed_CCE_preference"])
    count=sum(x["CCE_accepted_action_joined_AFTER_heuristic_computation"] is not None for x in predicted)
    matched=sum(x["matches_confirmed_CCE_when_available"] is True for x in predicted)
    abstain=sum(x["static_local_heuristic_chosen"] is None for x in predicted)
    result={
       "status":"TRAIN_ONLY_CHEAP_LOCAL_PROOF_HEURISTIC_AUDIT",
       "independent_task_roots":7,"states":14,
       "CCE_accepted_count":count,
       "static_heuristic_match_CCE_on_accepted":matched,
       "static_heuristic_unresolved_on_authorized":abstain,
       "static_proof_distance_or_closure_is_NOT_model_Qpi":True,
       "drawbacks":"assumes future task sources disclosed RAW; does not model actual B/R/other agent stochastic behavior or deadlines",
       "generator_easy_local_shortcut_risk":"ALL CCE-accepted preferences predictable by locally visible public Horn closure in current frozen synthetic panel" if matched==count else "SOME CCE outcomes not predictable locally",
       "not_a_real_world_privacy_legal_rule":True,
       "never_uses_Qstar_label_as_input":True,
       "never_modifies_Phase2_source_or_4B_checkpoint":True,
       "rows":predicted}
    (D/"STATIC_LOCAL_CLOSURE_AUDIT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_STATIC_LOCAL_PROOF_SHORTCUT",json.dumps(
       {k:v for k,v in result.items() if k!="rows"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
