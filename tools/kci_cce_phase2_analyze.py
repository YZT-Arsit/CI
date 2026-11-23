"""CCE Phase2 evidence audit: REAL frozen 4B Qpi vs terminal propagation.

NEVER uses bounded Qstar to choose a CCE or terminal-propagation label.
Qstar is joined only AFTER preference decisions are frozen, in a separate
audit key. All roots TRAIN; P211/P212/DG36 untouched.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_cce_phase2_preference import SEEDS,compare
SOURCE=Path(r"D:\research\kci_data_v4_phase2_20261122\artifacts\20261122_kci_data_v4_phase2_full_v1")
ACTIONS=("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")
ALLOW=frozenset(("P201","P202","P203","P205","P206","P207","P208"))
def sha(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1048576),b""):h.update(c)
    return h.hexdigest()
def jl(path):
    return [json.loads(s) for s in Path(path).open(encoding="utf8") if s.strip()]
def safe_qstar_audit_after_freeze():
    evidence={}
    with (SOURCE/"TRAIN_AND_DEV_EXACT_NATIVE_FUTURE_EVAL_ONLY.jsonl").open(encoding="utf8") as f:
        for line in f:
            # The native source mixes TRAIN/DEV rows. Filter on the raw line
            # BEFORE JSON decoding so that DEV and FINAL are never parsed.
            if not any(('"root_id":"'+root+'"') in line for root in ALLOW):
                continue
            if '"final_test"' in line or '"P211' in line or '"P212' in line:
                raise AssertionError("Unexpected sealed root in a train-only line")
            x=json.loads(line)
            assert x.get("split")=="train" and x.get("root_id") in ALLOW
            root=x["root_id"]
            assert root not in evidence
            branches=x["A_branches"]
            assert set(branches)=={"risky","safe"}
            native={}
            for category,b in branches.items():
                native[b["actual_A_action"]]={
                  "source_kind":category,
                  "bounded_CI_safe_completion_exists":b["any_CI_safe_business_completion"],
                  "minimum_safe_success_native_steps":b["best_bounded_safe_success_cost"],
                  "evidence_grade":"exact_within_horizon_native_not_Qpi",
                  "same_actual_A_state_sha256":x["A_same_state_sha256"]}
            assert set(native)==set(ACTIONS)
            winners=[(a,r) for a,r in native.items() if r["bounded_CI_safe_completion_exists"]]
            if len(winners)==1:
                preferred=winners[0][0]
                reason="bounded_native_success_exists_vs_none"
            elif len(winners)==2:
                costs=[(a,r["minimum_safe_success_native_steps"]) for a,r in winners]
                assert all(isinstance(n,int) for a,n in costs)
                if costs[0][1]!=costs[1][1]:
                    preferred=min(costs,key=lambda t:t[1])[0]
                    reason="bounded_native_both_success_cost_differ"
                else:
                    preferred=None;reason="bounded_native_equal_success_cost"
            else:
                preferred=None;reason="bounded_native_no_success"
            evidence[root]={
               "root":root,"source_kind":"native bounded exact B continuations after deterministic business pre-steps",
               "actual_A_same_state_sha256":x["A_same_state_sha256"],
               "A_native":native,
               "Qstar_preferred_A_if_defined":preferred,
               "Qstar_evidence_reason":reason,
               "NOT_Qpi":True,
               "never_used_to_construct_CCE_preference":True}
    assert set(evidence)==ALLOW
    return evidence

def aggregate(source,fork_file,output):
    if output.exists():raise FileExistsError(output)
    samples=jl(source)
    frozen=jl(fork_file)
    assert len(frozen)==14 and {x["root"] for x in frozen}==ALLOW
    by_cell={x["cell_id"]:x for x in frozen}
    assert len(by_cell)==len(frozen)
    assert len(samples)==14*2*len(SEEDS),len(samples)
    by_group=defaultdict(lambda:defaultdict(list))
    for s in samples:
        ident=s["cell_id"]
        assert ident in by_cell
        base=by_cell[ident]
        assert s["root"]==base["root"] and s["condition"]==base["condition"]
        assert s["start_A_state_hash"]==base["A_pre_state_sha256"]
        act=s["A_first_action"]
        assert act in ACTIONS
        assert s["A_after_state_hash"]==base["A_valid_candidates"][act]["post_A_native_hash"]
        assert s["A_immediate_verdict"]=="VALID" and not s["A_immediate_CI_violation"]
        assert s["is_actual_frozen_4B_Qpi_sample_not_oracle_Qstar"]
        assert s["seed"] in SEEDS
        assert s["native_transition_trace"][0]["native_action"]==act
        assert s["native_transition_trace"][0]["decision_source"]=="FORCED_SAME_STATE_A_CANDIDATE"
        assert s["native_transition_trace"][0]["native_verdict"]=="VALID"
        assert s["native_transition_trace"][0]["actual_recipient_ledgers"]
        assert s["model_action_count"]>=0
        assert not (s["ci_safe_success"] and (s["ci_violation"] or s["aborted"] or s["deadline_missed"] or s["unknown_or_unfinished"]))
        by_group[ident][act].append(s)
    decisions=[]
    model_calls=Counter()
    for ident,base in sorted(by_cell.items()):
        pair=by_group[ident]
        assert set(pair)==set(ACTIONS)
        two={}
        for a,rows in pair.items():
            byseed={r["seed"]:r for r in rows}
            assert len(rows)==len(byseed)==len(SEEDS)
            two[a]=[byseed[i] for i in SEEDS]
            for r in rows:
                model_calls.update(r["downstream_model_decision_roles"])
        result=compare(two)
        item={
          "cell_id":ident,"root":base["root"],"condition":base["condition"],
          "business_kind":base["business_kind"],
          "deadline_class":base["deadline_class"],
          "A_same_state_sha256":base["A_pre_state_sha256"],
          "A_current_action_verdicts":{a:base["A_valid_candidates"][a]["A_verdict"] for a in ACTIONS},
          "A_knowledge_delta":{a:base["A_valid_candidates"][a]["knowledge_delta"] for a in ACTIONS},
          "A_actual_received_counts":{a:base["A_valid_candidates"][a]["actual_delivery_count_delta"] for a in ACTIONS},
          "CCE_preference":result["preference"],
          "CCE_abstain_reason":result["abstain_reason"],
          "CCE_evidence_layer":result["evidence_layer"],
          "CCE_tests":result["tests"],
          "CCE_sample_branch_stats":result["branch_stats"],
          "CCE_protocol_priority":result["ci_first_comparison_priority"],
          "simple_terminal_propagation":result["scalar_terminal_propagation"],
          "Qstar_late_join_unused_for_labeling":None,
          "actual_policy_next_action_roles":{"A":dict(Counter(x for r in two[ACTIONS[0]] for x in r["downstream_model_decision_roles"])),
                                             "B":dict(Counter(x for r in two[ACTIONS[1]] for x in r["downstream_model_decision_roles"]))},
          "actual_model_Qpi_samples_not_exact_Qpi":True,
        }
        decisions.append(item)
    # At this point ALL CCE+terminal values are fixed; only now join bounded Qstar
    audit=safe_qstar_audit_after_freeze()
    for item in decisions:
        target=audit[item["root"]]
        assert target["actual_A_same_state_sha256"]==item["A_same_state_sha256"]
        if item["condition"]=="risky":
            item["Qstar_late_join_unused_for_labeling"]=target
        else:
            item["Qstar_late_join_unused_for_labeling"]={
              "reason":"native bounded source was computed in original unlicensed condition; no authorized-counterfactual bounded Qstar was supplied",
              "NOT_Qpi":True,"do_not_impute_Qstar":True}
        oracle=target["Qstar_preferred_A_if_defined"] if item["condition"]=="risky" else None
        for method,guess in (
            ("CCE",item["CCE_preference"]),
            ("terminal",item["simple_terminal_propagation"]["chosen_if_any"])):
            item[method+"_agrees_with_bounded_native_when_available"]=(
                None if oracle is None or guess is None else guess==oracle)
    assert len(decisions)==14
    count=Counter(x["CCE_evidence_layer"] or "ABSTAIN:"+str(x["CCE_abstain_reason"]) for x in decisions)
    stat=Counter(x["simple_terminal_propagation"]["chosen_if_any"] or "terminal_tie" for x in decisions)
    qpi=Counter()
    all_seeds=defaultdict(int)
    for s in samples:
        all_seeds[s["seed"]]+=1
        qpi.update({"actual_frozen_4B_total":1,
                    "CI_violations":int(s["ci_violation"]),
                    "CI_safe_success":int(s["ci_safe_success"]),
                    "safe_abort":int(s["aborted"] and not s["ci_violation"]),
                    "deadline_miss":int(s["deadline_missed"]),
                    "UNKNOWN_or_incomplete":int(s["unknown_or_unfinished"]),
                    "actual_model_downstream_choices":s["model_action_count"]})
    assert set(all_seeds)==set(SEEDS)
    out={
      "cell":"Actual frozen 4B Phase2 CCE same-state upstream A evidence",
      "root_count":len({x["root"] for x in decisions}),
      "distinct_knowledge_function_equivalences":[["P201","P206"],["P203","P208"]],
      "approx_distinct_boolean_function_groups":5,
      "same_A_state_count":len(decisions),
      "eligible_same_state_A_pairs":len(decisions),
      "full_native_model_Qpi_sampled_trajectories":len(samples),
      "n_model_K_per_A_action":len(SEEDS),
      "CI_first_provisional_preference_distribution":dict(count),
      "terminal_scalar_baseline_preferences":dict(stat),
      "CCE_accepted":sum(x["CCE_preference"] is not None for x in decisions),
      "CCE_abstained":sum(x["CCE_preference"] is None for x in decisions),
      "terminal_accepted_no_uncertainty_gate":sum(x["simple_terminal_propagation"]["chosen_if_any"] is not None for x in decisions),
      "terminal_given_direction_when_CCE_abstained":sum(x["CCE_preference"] is None and x["simple_terminal_propagation"]["chosen_if_any"] is not None for x in decisions),
      "terminal_wrong_when_Qstar_available":sum(x["terminal_agrees_with_bounded_native_when_available"] is False for x in decisions),
      "CCE_wrong_when_Qstar_available":sum(x["CCE_agrees_with_bounded_native_when_available"] is False for x in decisions),
      "all_Qpi_branch_outcomes":dict(qpi),
      "model_action_roles":dict(model_calls),
      "same_sample_seeds_each_A_branch":list(SEEDS),
      "source_raw_actual_4B_trajectories_SHA256":sha(source),
      "source_frozen_state_inputs_SHA256":sha(fork_file),
      "Qstar_separately_audited_only_late_join":True,
      "no_Qstar_labels_used_for_CCE_or_terminal_preference":True,
      "no_DG36_P211_P212_final_test_or_DEV_used":True,
      "human_real_world_privacy_norm_review_pending":True,
      "evidence_status":"paired_stochastic_evidence_with_explicit_abstention_not_exact_optimum",
      "clear_diagnostics":{
        "CCE_not_a_new_4B_policy_training_result":True,
        "ordinary_terminal_baseline_is_scalar_propagation_proxy_not_official_PrivAct_reproduction":True,
        "not_allow_ce_or_dpo_training_without_CCE_signal":True,
        "terminal_error_vs_native_bounded_only_mechanism_audit":True,
      }}
    output.mkdir(parents=True)
    (output/"CCE_PAIR_PREFERENCES.jsonl").write_text("".join(
        json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in decisions),encoding="utf8")
    (output/"CCE_MECHANISM_SUMMARY.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    (output/"NATIVE_BOUND_QSTAR_AUDIT_ONLY.json").write_text(
        json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_PHASE2_MECHANISM",json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--samples",type=Path,required=True)
    a.add_argument("--frozen-A",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    args=a.parse_args()
    aggregate(args.samples,args.frozen_A,args.output)
