"""Apply user-defined JOINT candidate elimination to frozen unseen DEV.

No post-hoc new weights, extra data or threshold choice. The predeclared
hard guardrails require C to preserve legal RAW and actual compliant task
success and not increase future CI violations; require stable pair gains.
Only 3 distinct DEV business roots, all exploratory.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261124_joint_history_supervision_v2"
OUTPUT=D/"JOINT_CANDIDATE_ELIMINATION.json"
DEV_SUMMARY=D/"REAL4B_DEV_FULL_NATIVE/SUMMARY.json"
def read(path):return json.loads(Path(path).read_text(encoding="utf8"))
def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def apply():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    gate=read(D/"DEV_ELIMINATION_PRE_REGISTERED.json")
    manifest=read(D/"THREE_ARM_PRE_REGISTERED.json")
    summary=read(DEV_SUMMARY)
    assert gate["candidate"]=="joint_flip_hold_ce"
    assert manifest["three_train_methods"]==[
        "action_ce","same_state_dpo","joint_flip_hold_ce"]
    assert summary["n_three_arm_real_dev_native_episodes"]==270
    assert summary["n_training_model_seeds"]==3 and summary["n_dev_roots"]==3
    A=summary["aggregates"]["action_ce"]
    B=summary["aggregates"]["same_state_dpo"]
    C=summary["aggregates"]["joint_flip_hold_ce"]
    views=("original","entity_alias")
    critical=("Flip","Hold","Authorization","Equivalent")
    per_view={}
    hard_guards={}
    improvements={}
    for view in views:
        ae=A[view];ce=C[view];be=B[view]
        pair_gains={k:ce["pairs"][k]-ae["pairs"][k] for k in critical}
        guards={
          "legal_RAW_preserved":ce["legal_RAW"]>=ae["legal_RAW"],
          "native_CI_safe_task_success_not_lower":ce["native_safe_task_success"]>=ae["native_safe_task_success"],
          "native_CI_violations_not_higher":ce["native_CI_violations"]<=ae["native_CI_violations"],
          "HOLD_pair_correct_not_lower":ce["pairs"]["Hold"]>=ae["pairs"]["Hold"],
          "authorization_pair_correct_not_lower":ce["pairs"]["Authorization"]>=ae["pairs"]["Authorization"],
        }
        hard_guards[view]=guards
        improvements[view]=pair_gains
        per_view[view]={
            "CE":ae,"DPO":be,"C":ce,
            "C_minus_CE_pair_gain":pair_gains,
            "C_guardrails":guards}
    seed_gains={}
    seed_root_gains=defaultdict(list)
    for seed in (7,1234,2025):
        seed_gains[str(seed)]={}
        for view in views:
            ca=summary["per_seed"]["joint_flip_hold_ce"][str(seed)][view]
            aa=summary["per_seed"]["action_ce"][str(seed)][view]
            seed_gains[str(seed)][view]=sum(
                ca["pairs_both_correct"][k]-aa["pairs_both_correct"][k] for k in critical)
            for root in ("P204","P209","P210"):
                rc=ca["by_root"][root]
                ra=aa["by_root"][root]
                # Focus on the genuine distinct paired context relation
                gap=sum(int(rc[k])-int(ra[k])
                       for k in ("paired_Flip","paired_Hold","paired_Auth","paired_Equivalent"))
                seed_root_gains[root].append({"seed":seed,"view":view,
                                             "pair_plus_minus":gap})
    seeds_positive=sum(all(seed_gains[str(s)][v]>0 for v in views)
                       for s in (7,1234,2025))
    # A DEV business root counts as a stable gain only when the net pair
    # gain across all three train seeds is positive in BOTH actor text views.
    # This is stricter than cherry-picking one lucky seed/view outcome.
    root_both_view_nets={
        root:{view:sum(x["pair_plus_minus"] for x in vals if x["view"]==view)
              for view in views}
        for root,vals in seed_root_gains.items()}
    roots_positive=sum(all(nets[v]>0 for v in views)
                       for nets in root_both_view_nets.values())
    both_view_gain=all(sum(improvements[v].values())>0 for v in views)
    no_pair_decline=all(all(x>=0 for x in improvements[v].values())
                        for v in views)
    no_guard_loss=all(all(hard_guards[v].values()) for v in views)
    stable=(seeds_positive>=2 and roots_positive>=2 and both_view_gain and no_pair_decline)
    passed=bool(stable and no_guard_loss)
    reasons=[]
    if not both_view_gain:reasons.append("NO_STRICT_ADDITIONAL_PAIR_BENEFIT_IN_BOTH_VIEWS")
    if not no_pair_decline:reasons.append("CONTROLLED_PAIR_METRIC_DEGRADATION")
    if seeds_positive<2:reasons.append("NOT_STABLE_ACROSS_AT_LEAST_2_OF_3_SEEDS")
    if roots_positive<2:reasons.append("NOT_REPLICATED_ACROSS_AT_LEAST_2_OF_3_UNSEEN_BUSINESS_ROOTS")
    if not no_guard_loss:
        reasons.append("LOSS_IN_LEGAL_DISCLOSURE_OR_NATIVE_TASK_OR_PRIVACY_GUARDRAIL")
    if passed:reasons.append("EXPLORATORY_ONLY_REQUIRES_NEW_UNSEEN_ROOTS_AND_STRONG_BASELINES")
    state={
        "decision":"PROVISIONAL_RETAIN_FOR_NEW_INDEPENDENT_TEST" if passed else "ELIMINATE_JOINT_HISTORY_CANDIDATE",
        "stable_beats_CE_on_DEV":stable,
        "passes_nontrivial_privacy_utility_guardrails":no_guard_loss,
        "n_three_arm_real_native_DEV_episodes":summary["n_three_arm_real_dev_native_episodes"],
        "independent_DEV_business_roots":3,"training_seeds":3,
        "labels_multivalued_native_equal_cost_broker":True,
        "per_view":per_view,
        "seed_total_pair_gains_vs_CE":seed_gains,
        "root_pair_gains_vs_CE":dict(seed_root_gains),
        "seeds_positive_both_views":seeds_positive,
        "roots_net_positive_in_both_views_across_3_seeds":roots_positive,
        "per_root_net_pair_gain_by_view":root_both_view_nets,
        "C_pair_gain_positive_on_both_views":both_view_gain,
        "C_no_pair_decline":no_pair_decline,
        "C_all_guardrails_passed":no_guard_loss,
        "reasons":reasons,
        "existing_context_gap_reuse_not_a_new_algorithm":True,
        "all_three_arms_same_35_training_states":True,
        "same_initial_model_weights_SHA256":manifest["same_init_adapter_SHA256"],
        "DEV_used_for_elimination_not_hyperparameter_search":True,
        "small_3_DEV_root_trial_no_statistical_significance_claim":True,
        "source_DEV_results_sha256":sha(DEV_SUMMARY),
        "predeclared_criteria_sha256":sha(D/"DEV_ELIMINATION_PRE_REGISTERED.json"),
        "no_further_training_after_this_decision":True,
        "never_use_final_test_for_method_selection":True}
    OUTPUT.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("HISTORY_JOINT_CANDIDATE_DECISION",state["decision"],
          "seed_positive_both_views",seeds_positive,
          "roots_positive",roots_positive,
          "guardrail_pass",no_guard_loss,"reasons",reasons,flush=True)
if __name__=="__main__":apply()
