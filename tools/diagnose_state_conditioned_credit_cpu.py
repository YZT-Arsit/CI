"""Read-only CPU examination of history-paired credit symmetry and Qpi records."""
from pathlib import Path
import json
import statistics
from collections import Counter,defaultdict
from diagnose_state_matched_cpu import BASE, OUT

def main():
    result_path=OUT/"CPU_CREDIT_DIAGNOSTIC.json"
    d=json.loads(result_path.read_text(encoding="utf-8"))
    sel=json.loads((BASE/"selection_v2"/"SELECTION.json").read_text(encoding="utf-8"))
    pairs={int(i):row for i,row in enumerate(sel["selected_pair_metadata"])}
    assert len(pairs)==8
    group=d["credit_mechanism"]["groups"]
    qpi=d["credit_mechanism"]["frozen_qpi_vs_realized_onpolicy_actions"]
    evidence={x["group"]:x for x in qpi}
    outcomes=[]
    total=Counter()
    for pid,record in sorted(pairs.items()):
        sides={g["related"]:g for g in group if g["pair_id"]==pid}
        assert set(sides)=={True,False}
        sensitive=bool(record["history_sensitive_qstar"])
        related=sides[True];unrelated=sides[False]
        supports=int(related["pCCE_supported"])+int(unrelated["pCCE_supported"])
        total["all_pairs"]+=1
        total["flip_pairs" if sensitive else "hold_pairs"]+=1
        total["flip_with_two_sided_credit" if sensitive and supports==2 else
              "flip_with_one_sided_credit" if sensitive and supports==1 else
              "flip_with_no_credit" if sensitive else
              "hold_with_two_sided_credit" if supports==2 else
              "hold_with_one_sided_credit" if supports==1 else
              "hold_with_no_credit"]+=1
        def preferred(q):
            return max(q,key=lambda k:q[k]["q"])
        qstar_u=record["native_qstar_unrelated"]
        qstar_r=record["native_qstar_related"]
        best_u=preferred(qstar_u)
        best_r=preferred(qstar_r)
        qstar_flip=(best_u!=best_r)
        observed_qpi={}
        for label,g in (("related",related),("unrelated",unrelated)):
            v=evidence.get(g["group"])
            if v is None:
                observed_qpi[label]={"available":False,"reason":"Rejected fork: branch-rollout estimates are not retained in 64 accepted proofs"}
            else:
                actions=v["frozen_policy_qpi_mean"]
                observed_qpi[label]={
                    "available":True,
                    "qpi_preferred":max(actions,key=actions.get),
                    "gap":v["frozen_policy_gap"],
                    "action_values":actions,
                    "teacher_signal_gate":v["frozen_policy_gate"],
                }
        outcomes.append({
            "pair_id":pid,"source_Qstar_history_sensitive":sensitive,
            "Qstar_top_action_unrelated":best_u,
            "Qstar_top_action_related":best_r,
            "Qstar_preference_flip":qstar_flip,
            "Qstar_max_within_state_margin":record["largest_within_state_qstar_gap"],
            "credit_coverage":{"unrelated":int(unrelated["pCCE_supported"]),
                               "related":int(related["pCCE_supported"])},
            "real_model_Qpi_on_supported_sides_only":observed_qpi,
        })
    summary={
        "pairs":outcomes,"counts":dict(total),
        "accepted_credit_records_on_related_histories":sum(8*x["credit_coverage"]["related"] for x in outcomes),
        "accepted_credit_records_on_unrelated_histories":sum(8*x["credit_coverage"]["unrelated"] for x in outcomes),
        "interpretation":{
            "pairwise_supervision":"The extra PCCE loss is per-state first-action additive credit, not a joint contrastive flip/hold objective across both related and unrelated states.",
            "rejected_branch_data_not_retained":"Group logs store rejection flags, not all raw Qpi branch samples. Their values cannot be reconstructed from saved accepted-credit proofs.",
            "Qstar_is_not_training_teacher":"Selection metadata with Qstar is used here only to name frozen flip/hold pairs."
        }
    }
    out=OUT/"CPU_HISTORY_PAIR_DIAGNOSTIC.json"
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    md=[
        "# History-paired PCCE credit diagnosis",
        "",
        "Counts: "+json.dumps(summary["counts"],ensure_ascii=False),
        "",
        f"Related-side accepted credits: {summary['accepted_credit_records_on_related_histories']}; unrelated-side: {summary['accepted_credit_records_on_unrelated_histories']}.",
        "",
        "| Pair | Qstar flip/hold | Unrelated credit | Related credit | Accepted Qpi preference |",
        "|---|---|---:|---:|---|",
    ]
    for x in outcomes:
        qpi_side=x["real_model_Qpi_on_supported_sides_only"]
        pref="/".join(f"{s}:{(qpi_side[s].get('qpi_preferred') or 'not-retained')}" for s in ("unrelated","related"))
        md.append(f"| {x['pair_id']} | {'Flip' if x['source_Qstar_history_sensitive'] else 'Hold'} | {x['credit_coverage']['unrelated']} | {x['credit_coverage']['related']} | {pref} |")
    md+=["","## Interpretation",
         "- The four preselected Qstar history-sensitive Flip pairs all have local PCCE credit on the related side only.",
         "- No Qstar Flip pair has *both sides* with accepted PCCE credit; terminal GRPO still trains on both sides.",
         "- Two Hold pairs have local credit on both sides and two have no accepted credit.",
         "- This asymmetry is a possible supervision-to-target mismatch, NOT an experimentally proven reason for weak DEV results.",
         "- PCCE is not a jointly supervised cross-history policy-consistency objective, and accepted credits are TRAIN-selected, not a generalization test.",
         "- Failed-Qpi branch estimates for the eight unsupported states are not preserved in this archived record; only acceptance/rejection flags remain."]
    (OUT/"CPU_HISTORY_PAIR_DIAGNOSTIC.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
