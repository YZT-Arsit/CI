"""Final root-aware evidence aggregation for the one evaluator development cell."""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts"/"20261122_history_evaluator_dev_v1"
RAW,VERIFY="SEND_RAW_R","VERIFY_PRIVATE_R"
def read_json(name):return json.loads((D/name).read_text(encoding="utf8"))
def read_jsonl(name):
    return [json.loads(s) for s in (D/name).read_text(encoding="utf8").splitlines() if s.strip()]
def main():
    native=read_jsonl("PAIR_STATES.jsonl")
    assert len(native)==56
    cases={(r["world"],r["branch"]):r for r in native}
    worlds=defaultdict(dict)
    for r in native:worlds[r["world"]][r["branch"]]=r
    assert len(worlds)==28 and all(set(x)=={"risky","safe"} for x in worlds.values())
    def two_way(by):
        return by["risky"]["aware_label"]==VERIFY and by["safe"]["aware_label"]==RAW
    def pair_meta(w,by):
        r=by["risky"]
        return {"world":w,"root":r["root"],"split":r["split"],"source_mode":r["source_mode"],
               "two_way":two_way(by)}
    wrows=[pair_meta(w,by) for w,by in worlds.items()]
    def native_stratum(xs):
        a=[w for w in xs if w["two_way"]]
        saves=[]
        for w in a:
            risk=worlds[w["world"]]["risky"]
            safe=worlds[w["world"]]["safe"]
            assert risk["aware"][RAW]["verdict"]=="VIOLATION"
            assert risk["aware"][VERIFY]["verdict"]=="VALID" and risk["aware"][VERIFY]["safe_success"]
            assert safe["aware"][RAW]["verdict"]=="VALID" and safe["aware"][RAW]["safe_success"]
            assert safe["aware"][VERIFY]["safe_success"]
            assert risk["blind_label"]==safe["blind_label"]==RAW
            saves.append(safe["aware"][VERIFY]["steps"]-safe["aware"][RAW]["steps"])
            assert saves[-1]>0
        return {
           "n_worlds":len(xs),"n_independent_roots":len({x["root"] for x in xs}),
           "two_way_worlds":len(a),
           "two_way_roots":sorted({x["root"] for x in a}),
           "two_way_independent_roots":len({x["root"] for x in a}),
           "safe_side_raw_steps_saved_distribution":dict(Counter(saves)),
        }
    native_strata={
       "all":native_stratum(wrows),
       "matched_only":native_stratum([x for x in wrows if x["source_mode"]=="matched"]),
       "train_matched":native_stratum([x for x in wrows if x["split"]=="train" and x["source_mode"]=="matched"]),
       "dev_matched":native_stratum([x for x in wrows if x["split"]=="dev" and x["source_mode"]=="matched"]),
       "exploratory_matched":native_stratum([x for x in wrows if x["split"]=="exploratory_test" and x["source_mode"]=="matched"]),
    }
    nonhistory=read_json("NONHISTORY_MATCH_AUDIT.json")
    assert native_strata["matched_only"]["two_way_worlds"]==6
    assert nonhistory["strict_two_way_nonhistory_identical"]==6
    truth={(r["world"],r["branch"]):r for r in native}
    def results_for(preds,cond):
        rows=[r for r in preds if cond(r)]
        grouped=defaultdict(dict)
        for r in rows:grouped[r["world"]][r["branch"]]=r
        assert all(set(g)=={"risky","safe"} for g in grouped.values())
        counts={}
        for method in ("visible_teacher","blind_teacher"):
            pr=[(r,r["predictions"][method]) for r in rows]
            strict=[pair for pair in grouped.values() if pair["risky"]["true_label"]==VERIFY and pair["safe"]["true_label"]==RAW]
            counts[method]={
              "n_state_rows":len(rows),
              "correct_states":sum(r["true_label"]==p["balanced"] for r,p in pr),
              "risk_states":sum(r["true_label"]==VERIFY for r,p in pr),
              "risk_states_correct_verify":sum(r["true_label"]==VERIFY and p["balanced"]==VERIFY for r,p in pr),
              "safe_states":sum(r["true_label"]==RAW for r,p in pr),
              "safe_states_correct_raw":sum(r["true_label"]==RAW and p["balanced"]==RAW for r,p in pr),
              "n_paired_worlds":len(grouped),
              "n_two_way_worlds":len(strict),
              "n_independent_roots":len({r["root"] for r in rows}),
              "n_two_way_independent_roots":len({b["risky"]["root"] for b in strict}),
              "two_way_both_correct":sum(all(b[side]["predictions"][method]["balanced"]==b[side]["true_label"]
                  for side in ("safe","risky")) for b in strict),
              "two_way_semantic_action_flips":sum(b["risky"]["predictions"][method]["balanced"]!=
                  b["safe"]["predictions"][method]["balanced"] for b in strict),
              "order_swap_changes_semantic_action":sum(p["original"]!=p["swapped"] for r,p in pr),
              "balanced_action_counts":dict(Counter(p["balanced"] for r,p in pr))
            }
        return counts
    sources={
       "dpo32":"gpu_eval_dpo_s7/DECISIONS.jsonl",
       "ce32":"gpu_eval_ce_s7/DECISIONS.jsonl",
       "ce16":"gpu_eval_strict16_ce_s7/DECISIONS.jsonl"
    }
    strata={
      "train_all_32":lambda r:r["split"]=="train",
      "train_strict_all_16":lambda r:r["split"]=="train" and r["root"] in ("DG01","DG05"),
      "train_strict_matched_8":lambda r:r["split"]=="train" and r["root"] in ("DG01","DG05")
                  and truth[(r["world"],r["branch"])]["source_mode"]=="matched",
      "dev_all_8":lambda r:r["split"]=="dev",
      "exploratory_all_16":lambda r:r["split"]=="exploratory_test",
      "exploratory_DG18_matched_4":lambda r:r["split"]=="exploratory_test" and r["root"]=="DG18"
                  and truth[(r["world"],r["branch"])]["source_mode"]=="matched",
      "exploratory_DG18_all_8":lambda r:r["split"]=="exploratory_test" and r["root"]=="DG18"
    }
    methods={}
    for name,filename in sources.items():
        data=read_jsonl(filename)
        assert len(data)==56
        for r in data:
            assert r["true_label"]==truth[(r["world"],r["branch"])]["aware_label"]
        methods[name]={key:results_for(data,cond) for key,cond in strata.items()}
    assert methods["dpo32"]["train_strict_all_16"]["visible_teacher"]["two_way_both_correct"]==0
    assert methods["ce32"]["train_strict_all_16"]["visible_teacher"]["two_way_both_correct"]==0
    assert methods["ce16"]["train_strict_all_16"]["visible_teacher"]["two_way_both_correct"]==0
    sources_hashed={name:sha256((D/fname).read_bytes()).hexdigest() for name,fname in sources.items()}
    summary={
       "cell":"DEV history-aware native evaluator vs receipt-blind native evaluator, followed by matched 4B action learning",
       "status":"CLOSED_WITH_NEGATIVE_POLICY_ACTION_RESULT",
       "native_evaluator":native_strata,
       "confound_control":{
           "strict_two_way_worlds_with_identical_nonhistory_prompt":6,
           "n_all_two_way_worlds":12,
           "tradeoff_excluded_from_strict_pure_history_claims":6,
           "only_changed_non_history_field_in_tradeoff":"Remaining work budget"},
       "train_teacher_counts":read_json("DATA_MANIFEST.json")["train_labels"],
       "strict16_teacher_counts":read_json("STRICT16_MANIFEST.json")["teacher_labels"],
       "pure_strict8_teacher_counts":read_json("STRICT8_MANIFEST.json")["labels"],
       "models":methods,
       "source_decisions_files":sources,
       "decisions_sha256":sources_hashed,
       "source_native_rows":"PAIR_STATES.jsonl",
       "no_blind_new_root_claim":True,
       "action_representation_gate_failed":True,
       "capacity_attribution_established":False,
       "suggested_research_decision":"Native evaluator passes paired-history supervision, but trained 4B categorical action policy does not fit exact two-sided states. Stop scaling or extending algorithm; next repair action encoding / order invariance using the already frozen matched pure-history pairs. Do not infer capacity failure.",
       "independent_root_count_is_not_derived_world_count":True,
    }
    dest=D/"FINAL_DEV_CELL_SUMMARY.json"
    assert not dest.exists()
    dest.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    compact={
      "native":native_strata,
      "model_strict_train_pure_matched":{m:methods[m]["train_strict_matched_8"] for m in methods},
      "model_exploratory_DG18_pure_matched":{m:methods[m]["exploratory_DG18_matched_4"] for m in methods},
      "model_full_train":{m:methods[m]["train_all_32"] for m in methods}
    }
    print(json.dumps(compact,ensure_ascii=False,indent=2)[:22000])
if __name__=="__main__":main()
