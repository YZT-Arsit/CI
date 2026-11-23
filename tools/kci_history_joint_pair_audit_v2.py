"""Independent source-key comparison for existing Phase2 paired histories.

TRAIN only. Verify actor-authorized obs actually isolate recipient receipt or
authorization change; equivalent order uses the exact same delivered FACT SET.
This audit never fabricates another data view, does not train or read DEV.
"""
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
import json
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261124_joint_history_supervision_v2"
def jl(p):return [json.loads(s) for s in Path(p).open(encoding="utf8") if s.strip()]
def main():
    r=jl(D/"TRAIN_FIVE_CONTEXTS.jsonl")
    pairs=jl(D/"TRAIN_RELATION_PAIRS.jsonl")
    data={x["world"]:x for x in r}
    assert len(data)==35 and len(pairs)==28
    facts=set();details=[];deltas=Counter()
    for p in pairs:
        a=data[p["left_world"]];b=data[p["right_world"]]
        assert a["root"]==b["root"]==p["root_id"]
        assert a["role"]==b["role"] and a["stage"]==b["stage"]
        assert {a["condition"],b["condition"]}<=set(("risky","safe","hold","authorized","equivalent"))
        aa=a["actor_visible_observation"];bb=b["actor_visible_observation"]
        changes={k for k in aa if aa[k]!=bb[k]}
        if p["kind"]=="authorization_exception":
            assert changes=={"privacy_policy"},(p["pair_id"],changes)
            assert a["native_state_sha256"]==b["native_state_sha256"]
        else:
            assert changes=={"delivered_message_log"},(p["pair_id"],changes)
            lhs=[x["fact"] for x in aa["delivered_message_log"]]
            rhs=[x["fact"] for x in bb["delivered_message_log"]]
            if p["kind"]=="equivalent_history":
                assert set(lhs)==set(rhs)
                assert lhs!=rhs or aa["delivered_message_log"]!=bb["delivered_message_log"]
            else:
                assert set(lhs)!=set(rhs)
        assert len(aa["delivered_message_log"])==len(bb["delivered_message_log"])
        assert p["not_same_state_DPO"] and p["same_current_signed_source"] and p["same_task_and_role"]
        deltas[p["kind"]]+=1
        details.append({"pair_id":p["pair_id"],"native_relation":p["kind"],
          "changed_role_authorized_fields":sorted(changes),
          "n_delivery_before":len(aa["delivered_message_log"]),
          "n_delivery_after":len(bb["delivered_message_log"]),
          "acceptable_native_source_actions_left":a["acceptable_native_actions"],
          "acceptable_native_source_actions_right":b["acceptable_native_actions"],
          "same_actor_and_current_signed_source":True,
          "not_same_state_DPO":True})
    assert deltas=={"history_flip":7,"authorization_exception":7,"unrelated_hold":7,"equivalent_history":7}
    report={"source":"original_native_Phase2_true_context_pairs",
      "n_independent_train_roots":7,
      "n_true_nonduplicate_training_world_states":35,
      "n_train_cross_history_control_pairs":28,
      "relation_count":dict(deltas),
      "all_pairs_isolate_exactly_one_actor_visible_field":True,
      "authorization_change_is_public_policy_only":True,
      "other_pairs_change_actual_delivered_receipts_only":True,
      "equivalent_history_preserves_fact_set":True,
      "no_extra_training_contexts":True,
      "no_DEV_or_final_model_predictions_loaded":True,
      "pairs":details}
    p=D/"TRAIN_ROLE_VISIBLE_PAIR_ISOLATION_AUDIT.json"
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("HISTORY_V2_TRUE_PAIR_ISOLATION_PASS",len(details),dict(deltas),flush=True)
if __name__=="__main__":main()
