"""Post-hoc descriptive audit of terminal-credit instability and downstream events.

This does NOT change the CCE preregistered rule, labels, or model outputs.
Leave-one-seed-out is only a stability diagnostic (not confidence proof).
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import json
from kci_cce_phase2_preference import SEEDS,TERMINAL_PREFERENCE_EPSILON,terminal_return
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1"
SAMPLE=D/"POLICY_SAMPLED_4B_K8/SAMPLED_ACTUAL_FROZEN_4B_TRAJECTORIES.jsonl"
PREF=D/"CCE_ANALYSIS_K8/CCE_PAIR_PREFERENCES.jsonl"
def jl(p):return [json.loads(x) for x in p.read_text(encoding="utf8").splitlines() if x.strip()]
def scalar_pref(a,b):
    diff=sum(map(terminal_return,a))/len(a)-sum(map(terminal_return,b))/len(b)
    return "A_SEND_PACKAGE_A" if diff>TERMINAL_PREFERENCE_EPSILON else "A_SEND_PACKAGE_B" if diff < -TERMINAL_PREFERENCE_EPSILON else None
def main():
    traces=jl(SAMPLE);analysis=jl(PREF)
    qstar=json.loads((D/"CCE_ANALYSIS_K8/NATIVE_BOUND_QSTAR_AUDIT_ONLY.json").read_text(encoding="utf8"))
    assert len(traces)==224 and len(analysis)==14
    groups=defaultdict(lambda:defaultdict(dict))
    bykind=defaultdict(Counter)
    roles=Counter()
    for x in traces:
        key=(x["cell_id"],x["condition"])
        assert x["seed"] not in groups[key][x["A_first_action"]]
        groups[key][x["A_first_action"]][x["seed"]]=x
        ty=qstar[x["root"]]["A_native"][x["A_first_action"]]["source_kind"]+"_package"
        assert ty in ("risky_package","safe_package")
        bykind[(x["condition"],ty)].update({
          "trajectories":1,"CI_violations":int(x["ci_violation"]),
          "CI_safe_success":int(x["ci_safe_success"]),
          "safe_abort":int(x["aborted"] and not x["ci_violation"]),
          "deadline_missed":int(x["deadline_missed"])})
        roles.update(x["downstream_model_decision_roles"])
    records=[];unstable=Counter()
    for x in analysis:
        key=(x["cell_id"],x["condition"])
        pair=groups[key];A=pair["A_SEND_PACKAGE_A"];B=pair["A_SEND_PACKAGE_B"]
        assert set(A)==set(B)==set(SEEDS)
        full=scalar_pref([A[i] for i in SEEDS],[B[i] for i in SEEDS])
        assert full==x["simple_terminal_propagation"]["chosen_if_any"]
        jack=[]
        half=[]
        for seed in SEEDS:
            keep=[j for j in SEEDS if j!=seed]
            jack.append(scalar_pref([A[j] for j in keep],[B[j] for j in keep]))
        for keep in (SEEDS[:4],SEEDS[4:]):
            half.append(scalar_pref([A[j] for j in keep],[B[j] for j in keep]))
        stable=all(direction==full for direction in jack)
        loose=all(direction==full for direction in half)
        unstable.update({"terminal_full_has_direction":int(full is not None),
                         "terminal_leave_one_out_changes":int(not stable),
                         "terminal_half_split_changes":int(not loose),
                         "CI_risk_abstained_with_terminal_direction":
                             int(x["CCE_abstain_reason"]=="unresolved_nonzero_future_CI_risk_difference" and full is not None)})
        records.append({"cell_id":x["cell_id"],"root":x["root"],"condition":x["condition"],
                        "terminal_all_K8":full,
                        "terminal_jackknife_K7":jack,
                        "terminal_two_predefined_K4_halves":half,
                        "jackknife_preserves_full":stable,
                        "half_splits_preserve_full":loose,
                        "CCE_abstain_reason":x["CCE_abstain_reason"],
                        "CCE_is_accepted":x["CCE_preference"] is not None,
                        "native_Qstar_agreement_when_available":
                            x["terminal_agrees_with_bounded_native_when_available"],
                        "K4_K7_stability_is_posthoc_diagnostic_only":True})
    summary={
      "n_cells":len(records),
      "terminal_directions_K8":unstable["terminal_full_has_direction"],
      "n_jackknife_changed_including_tie":unstable["terminal_leave_one_out_changes"],
      "n_half_split_changed_including_tie":unstable["terminal_half_split_changes"],
      "CI_risk_unresolved_but_scalar_direction_chosen":unstable["CI_risk_abstained_with_terminal_direction"],
      "not_any_observed_scalar_direction_wrong_against_bounded_Qstar_on_unlicensed_train_roots":all(
          x["terminal_agrees_with_bounded_native_when_available"] is not False for x in analysis),
      "scalar_wrong_with_available_Qstar_count":sum(x["terminal_agrees_with_bounded_native_when_available"] is False for x in analysis),
      "outcomes_by_train_condition_and_package":{str(k):dict(v) for k,v in bykind.items()},
      "all_downstream_model_roles":dict(roles),
      "recorded_native_transition_events":sum(len(x["native_transition_trace"]) for x in traces),
      "posthoc_variation_not_CCE_decision_rule":True,
      "no_DG36_or_final_data_loaded":True}
    out=D/"TERMINAL_PROPAGATION_STABILITY.json"
    assert not out.exists()
    out.write_text(json.dumps({"summary":summary,"cells":records},ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_TERMINAL_STABILITY",json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":main()
