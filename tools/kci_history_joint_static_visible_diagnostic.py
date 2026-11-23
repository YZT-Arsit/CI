"""Visible-only static closure diagnostic, NOT another trained arm.

No hidden native state or native Qstar/Qpi for prediction; use only the
actor-visible delivered messages, signed current source, inference rules and
explicit authorization. Evaluate only AFTER all nine LLM arms are frozen.
This tests whether the user's feared short local rule already solves release.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.diverse_ci_branch_env import SLOT_LABELS
D=ROOT/"artifacts/20261124_joint_history_supervision_v2"
P=D/"VISIBLE_ONLY_STATIC_CLOSURE_DIAGNOSTIC.json"
F=D/"DEV_FIVE_CONTEXTS.jsonl"
def closure(public_rules,initial):
    facts=set(initial)
    for _ in range(len(public_rules)+2):
        newer=set(facts)
        for rule in public_rules:
            if set(rule["requires"])<=facts:
                newer.add(rule["entails"])
        if newer==facts:break
        facts=newer
    return facts
def local_visible_policy(ob):
    opts=ob["legal_actions"]
    assert "SEND_RAW_TARGET" in opts and "VERIFY_PRIVATE_TARGET" in opts
    ledger=ob["delivered_message_log"]
    base={SLOT_LABELS.get(x["fact"],x["fact"]) for x in ledger}
    new=ob["signed_source_tool_record"]
    assert new and new["signed"] and new["slot"]
    before=closure(ob["public_inference_rules"],base)
    after=closure(ob["public_inference_rules"],base|{new["slot"]})
    allowed=("A purpose-scoped authorization explicitly permits" in ob["privacy_policy"])
    forbidden=("No authorization exception is granted" in ob["privacy_policy"])
    assert allowed!=forbidden
    newly_infers_protected="protected association" in after and "protected association" not in before
    if newly_infers_protected and not allowed:
        return "VERIFY_PRIVATE_TARGET",{"would_newly_infer_protected":True,
            "authorized":False,"public_inference_before":sorted(before),
            "public_inference_after":sorted(after)}
    return "SEND_RAW_TARGET",{"would_newly_infer_protected":newly_infers_protected,
        "authorized":allowed,"public_inference_before":sorted(before),
        "public_inference_after":sorted(after)}
def main():
    if P.exists():raise FileExistsError(P)
    # Do not even open the DEV file until all matched models are frozen.
    for arm in ("action_ce","same_state_dpo","joint_flip_hold_ce"):
        for seed in (7,1234,2025):
            assert (D/f"train_{arm}_seed{seed}/TRAINING.json").exists()
    rows=[json.loads(x) for x in F.open(encoding="utf8") if x.strip()]
    assert len(rows)==15 and {x["split"] for x in rows}=={"dev"}
    records=[]
    for row in rows:
        choice,evidence=local_visible_policy(row["actor_visible_observation"])
        records.append({"root":row["root"],"condition":row["condition"],
            "static_closure_prediction":choice,
            "native_acceptable_evaluator_only":row["acceptable_native_actions"],
            "correct":choice in row["acceptable_native_actions"],
            "used_only_actor_visible_state":True,
            "visible_reasons":evidence})
    p={
       "status":"DEV_STATIC_VISIBLE_ONLY_DIAGNOSTIC_NOT_FOURTH_TRAIN_ARM",
       "predictor_uses_only_actor_visible_receipts_rules_source_authorization":True,
       "predictor_has_no_native_Qstar_or_future_reward":True,
       "n_dev_true_states":len(rows),
       "state_action_set_correct":sum(x["correct"] for x in records),
       "n_independent_dev_roots":3,
       "records":records,
       "possible_shortcut_warning":"A perfect static local closure predictor implies the current paired release problem does not require long-horizon learning",
       "not_the_main_three_arm_optimizer_comparison":True}
    P.write_text(json.dumps(p,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("JOINT_HISTORY_VISIBLE_STATIC_BASELINE",p["state_action_set_correct"],"/",p["n_dev_true_states"],flush=True)
if __name__=="__main__":main()
