"""Strict matching of original K8 frozen 4B vs complete-flow SFT K8.

Pair by SAME (root,context,pre-A state,first legal A action,seed).
No modeled Qstar data, no reranking, no test/dev input.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1/POLICY_SAMPLED_4B_K8/SAMPLED_ACTUAL_FROZEN_4B_TRAJECTORIES.jsonl"
AFTER=ROOT/"artifacts/20261124_cce_phase2_post_sft_v1/POST_SFT_FROZEN_ADAPTIVE/ACTUAL_POST_SFT_4B_NATIVE_ROLLOUTS.jsonl"
OUT=ROOT/"artifacts/20261124_cce_phase2_post_sft_v1/BEFORE_AFTER_MATCHED_K8.json"
SEEDS={7,1234,2025,42,314,2718,1618,99}
def jl(p):return [json.loads(s) for s in p.open(encoding="utf8") if s.strip()]
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def key(r):
    return (r["cell_id"],r["A_first_action"],r["seed"])
def stats(rows):
    aborts=Counter(e["stage"] for r in rows for e in r["native_transition_trace"] if e["native_action"]=="ABORT")
    completed=[r for r in rows if r["ci_safe_success"]]
    return {"n":len(rows),"compliant_business_completed":len(completed),
            "future_CI_violations":sum(r["ci_violation"] for r in rows),
            "safe_aborts":sum(r["aborted"] for r in rows),
            "deadline_misses":sum(r["deadline_missed"] for r in rows),
            "unknown_or_unfinished":sum(r["unknown_or_unfinished"] for r in rows),
            "total_real_model_downstream_decisions":sum(r["model_action_count"] for r in rows),
            "total_native_steps":sum(r["native_total_steps"] for r in rows),
            "mean_native_cost_among_successes":(sum(r["native_total_steps"] for r in completed)/len(completed) if completed else None),
            "actual_ABORT_by_stage":dict(aborts)}
def main():
    if OUT.exists():raise FileExistsError(OUT)
    old=jl(ORIGINAL)
    later=[x for x in jl(AFTER) if x["seed"] in SEEDS]
    assert len(old)==len(later)==224
    assert len({key(x) for x in old})==224
    assert len({key(x) for x in later})==224
    a={key(x):x for x in old};b={key(x):x for x in later}
    assert set(a)==set(b)
    for ident,earlier in a.items():
        next_=b[ident]
        assert earlier["start_A_state_hash"]==next_["start_A_state_hash"]
        assert earlier["A_after_state_hash"]==next_["A_after_state_hash"]
        assert earlier["A_immediate_verdict"]==next_["A_immediate_verdict"]=="VALID"
        assert not earlier["A_immediate_CI_violation"] and not next_["A_immediate_CI_violation"]
        assert earlier["root"]==next_["root"]
    original=stats(old);after=stats(later)
    paired=Counter()
    for key_,old_row in a.items():
        new=b[key_]
        if not old_row["ci_safe_success"] and new["ci_safe_success"]:paired["newly_CI_safe_complete"]+=1
        if old_row["ci_safe_success"] and not new["ci_safe_success"]:paired["lost_CI_safe_completion"]+=1
        if not old_row["ci_violation"] and new["ci_violation"]:paired["new_future_CI_violations"]+=1
        if old_row["ci_violation"] and not new["ci_violation"]:paired["avoided_old_future_CI_violation"]+=1
        if old_row["aborted"] and not new["aborted"]:paired["old_ABORT_now_nonabort"]+=1
    roots={}
    for root in ("P201","P202","P203","P205","P206","P207","P208"):
        o=[x for x in old if x["root"]==root]
        n=[x for x in later if x["root"]==root]
        assert len(o)==len(n)==32
        roots[root]={"before":stats(o),"after":stats(n)}
    outcome={
      "result":"MATCHED_SAME_224_K8_UPSTREAM_STATES_4B_TEAM_BEFORE_AFTER",
      "train_roots":7,"same_A_states":14,
      "SAME_semantic_current_A_native_knowledge_after_fork":True,
      "exactly_same_A_action_and_sample_seed_per_pair":True,
      "K8_seed_replay_same":True,
      "historic_shared_policy_joint3_vs_new_fresh_native_success_witness_SFT":True,
      "not_equal_training_initialization_or_train_budget_between_historical_joint3_and_new_SFT":True,
      "before":original,"after":after,"paired_change_counts":dict(paired),
      "per_business_root":roots,
      "old_4B_trace_sha256":sha(ORIGINAL),
      "new_4B_trace_sha256":sha(AFTER),
      "no_DEV_or_final_test_data_in_analysis":True,
      "not_causal_effect_of_CCE_or_training_A_preference":True}
    OUT.write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_SFT_MATCHED_K8",json.dumps({k:v for k,v in outcome.items() if k!="per_business_root"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
