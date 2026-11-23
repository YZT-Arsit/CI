"""Counterfactual control: apply ORIGINAL pre-SFT K8 CCE rule unchanged to NEW 4B K8.

This is a posthoc diagnostic that isolates *policy capability* from the
new conservative K16/K32 rule. It DOES NOT change model training labels or
overwrite any pre-registered adaptive result.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_cce_phase2_preference import SEEDS,compare
B=ROOT/"artifacts/20261124_cce_phase2_post_sft_v1"
D=B/"POST_SFT_FROZEN_ADAPTIVE/ACTUAL_POST_SFT_4B_NATIVE_ROLLOUTS.jsonl"
TARGET=B/"ORIGINAL_K8_RULE_REAPPLIED_TO_POST_SFT.json"
def main():
    if TARGET.exists():raise FileExistsError(TARGET)
    groups=defaultdict(lambda:defaultdict(dict))
    for l in D.open(encoding="utf8"):
        r=json.loads(l)
        if r["seed"] not in SEEDS:continue
        assert r["seed"] not in groups[r["cell_id"]][r["A_first_action"]]
        groups[r["cell_id"]][r["A_first_action"]][r["seed"]]=r
    assert len(groups)==14
    answers=[]
    for ident,actions in sorted(groups.items()):
        assert set(actions)=={"A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"}
        result=compare({a:[byseed[s] for s in SEEDS] for a,byseed in actions.items()})
        answers.append({"cell_id":ident,
            "original_K8_CCE_preference_on_new_SFT_frozen_4B":result["preference"],
            "original_K8_CCE_abstain_reason":result["abstain_reason"],
            "original_K8_CCE_evidence_layer":result["evidence_layer"],
            "terminal_scalar_K8":result["scalar_terminal_propagation"]["chosen_if_any"],
            "original_rule_fully_unchanged":True})
    stats={
      "comparison":"Reapply original CCE 20261123 K8 to 20261124 new frozen team",
      "n_same_A_states":14,"n_4B_native_K8_trajectories":224,
      "n_orig_K8_signal_accepted_after_SFT":sum(x["original_K8_CCE_preference_on_new_SFT_frozen_4B"] is not None for x in answers),
      "n_orig_K8_signal_abstained_after_SFT":sum(x["original_K8_CCE_preference_on_new_SFT_frozen_4B"] is None for x in answers),
      "old_K8_baseline_accept_before_SFT":0,
      "same_old_K8_paired_seeds":list(SEEDS),
      "no_rule_or_threshold_change_in_same_budget_comparison":True,
      "new_adaptive_protocol_not_replaced_or_selected":True,
      "no_training_from_this_posthoc_replay":True,
      "rows":answers}
    TARGET.write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("ORIGINAL_K8_RULE_REPLAY",json.dumps({k:v for k,v in stats.items() if k!="rows"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
