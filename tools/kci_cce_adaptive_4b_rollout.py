"""Adaptive K8->K16->K32 CCE on a frozen, shared 4B executor.

Same native A state, two *currently legal* actual A deliveries, genuine
native team continuations. Uses ONLY frozen model for each post-A action,
with no normative reward or Qstar entering actor input. Pre-registered
paired-seed blocks and conservative two-sided Bonferroni sign evidence.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
from hashlib import sha256
import argparse,json,time,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_cce_phase2_frozen4b_rollout import Frozen4B,one_trajectory,sha as hashfile
from kci_cce_adaptive_protocol import (
  ACTION_A,ACTION_B,ACTIONS,SEEDS32,BUDGETS,OUT,ROOT as FILE_ROOT,
  summarize_pair,should_continue,sha,ALPHA_CONFIRM)
STUDY_SOURCE=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1/FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl"
def jl(p):
    return [json.loads(s) for s in Path(p).open(encoding="utf8") if s.strip()]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--adapter",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source",type=Path,default=STUDY_SOURCE)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    prereg=ROOT/"artifacts/20261124_cce_phase2_post_sft_v1/PREREGISTERED_CCE_K8_ADAPTIVE.json"
    contract=json.loads(prereg.read_text(encoding="utf8"))
    assert tuple(contract["stage2_seeds32"])==SEEDS32
    assert tuple(contract["adaptive_budget_looks"])==BUDGETS
    assert contract["per_test_confirm_alpha"]==ALPHA_CONFIRM
    assert contract["source_same_state_A_forks_SHA256"]==sha(args.source)
    rows=jl(args.source)
    assert len(rows)==14
    assert {r["root"] for r in rows}=={
      "P201","P202","P203","P205","P206","P207","P208"}
    assert all(r["current_A_both_actions_native_CI_legal"] for r in rows)
    adapter=Path(args.adapter)/"adapter_model.safetensors"
    assert adapter.is_file()
    model_sha=hashfile(adapter)
    hf=Frozen4B(args.base,args.adapter)
    assert hf.hf.model is not None
    start=time.monotonic()
    args.output.mkdir(parents=True)
    cells=[];traces=[];accepted=Counter();abstained=Counter();kdist=Counter()
    with (args.output/"ACTUAL_POST_SFT_4B_NATIVE_ROLLOUTS.jsonl").open("x",encoding="utf8") as f:
        for row in rows:
            cell=row["cell_id"]
            sampled={A:[] for A in ACTIONS}
            looks=[]
            last_n=0
            for n in BUDGETS:
                for ix in range(last_n,n):
                    seed=SEEDS32[ix]
                    for act in ACTIONS:
                        result=one_trajectory(row,act,seed,hf,"single_shared_full_flow_SFT_4B")
                        assert result["A_immediate_verdict"]=="VALID" and not result["A_immediate_CI_violation"]
                        assert result["start_A_state_hash"]==row["A_pre_state_sha256"]
                        assert result["A_after_state_hash"]==row["A_valid_candidates"][act]["post_A_native_hash"]
                        result["policy_adapter_sha256"]=model_sha
                        result["sampling_protocol"]="independent_seed_blocks_sequential_RNG_each_trajectory"
                        f.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+"\n");f.flush()
                        sampled[act].append(result);traces.append(result)
                    print("CCE_SFT_4B_TRAJECTORY_PAIR",cell,seed,
                          {"A":[sampled[A][-1]["ci_violation"],sampled[A][-1]["ci_safe_success"],
                                sampled[A][-1]["aborted"],sampled[A][-1]["deadline_missed"]]
                           for A in ACTIONS},flush=True)
                last_n=n
                evidence=summarize_pair(sampled[ACTION_A],sampled[ACTION_B],n)
                actual=evidence["current_look"]
                cont,reason=should_continue(actual,n)
                looks.append({"K":n,"evidence":evidence,
                    "continue_to_more_samples":cont,"stop_reason":reason,
                    "model_candidate_forward_calls_so_far":hf.model_forwards,
                    "model_score_cache_hits_so_far":hf.cached_score_lookups})
                print("CCE_SFT_ADAPTIVE_LOOK",cell,"K",n,
                    "decision",actual["decision"],"confirmed",actual["confirmed"],
                    "evidence_status",actual["status"],
                    "risk",[(a,actual["outcomes"][a]["CI_violations"]) for a in ACTIONS],
                    "success",[(a,actual["outcomes"][a]["CI_safe_success"]) for a in ACTIONS],
                    "stop",reason,flush=True)
                if not cont:break
            final=looks[-1]["evidence"]["current_look"]
            kdist[len(sampled[ACTION_A])]+=1
            if final["confirmed"]:accepted[final["status"]]+=1
            else:abstained[final["status"]]+=1
            cells.append({
                "cell_id":cell,"root":row["root"],"condition":row["condition"],
                "A_pre_state_sha256":row["A_pre_state_sha256"],
                "A_action_current_CI_verdicts":{a:row["A_valid_candidates"][a]["A_verdict"] for a in ACTIONS},
                "post_A_actual_knowledge_delta":{a:row["A_valid_candidates"][a]["knowledge_delta"] for a in ACTIONS},
                "look_history":looks,
                "K_final_per_action":len(sampled[ACTION_A]),
                "confirmed_CCE_preference":final["decision"] if final["confirmed"] else None,
                "final_evidence_status":final["status"],
                "K8_terminal_scalar_propagation":looks[0]["evidence"]["terminal_K8"],
                "final_terminal_scalar_propagation":looks[-1]["evidence"]["terminal_current_n"],
                "no_native_Qstar_used_in_preference_generation":True})
    assert len(cells)==14
    n_complete=len(traces)
    meta={
       "study":"KCI_CCE_Phase2_post_complete_flow_4B_SFT",
       "same_fixed_A_states_before_after":True,
       "n_independent_train_business_roots":7,
       "n_same_upstream_A_states":len(cells),
       "n_real_frozen_4B_native_trajectories":n_complete,
       "K_per_action_distribution":dict(kdist),
       "first_eight_seeds_identical_to_original_baseline":True,
       "all_paired_seeds_by_look":list(SEEDS32),
       "model_adapter_sha256":model_sha,
       "policy_adapter":args.adapter,
       "base_model":args.base,
       "same_one_shared_policy_for_all_downstream_roles":True,
       "trained_with_A_SEND_preference":False,
       "adapter_not_updated_during_rollout":True,
       "actual_native_steps":sum(x["native_total_steps"] for x in traces),
       "actual_native_transition_events":sum(len(x["native_transition_trace"]) for x in traces),
       "actual_downstream_model_decisions":sum(x["model_action_count"] for x in traces),
       "CI_violations":sum(x["ci_violation"] for x in traces),
       "CI_safe_task_successes":sum(x["ci_safe_success"] for x in traces),
       "safe_aborts":sum(x["aborted"] and not x["ci_violation"] for x in traces),
       "deadline_misses":sum(x["deadline_missed"] for x in traces),
       "UNKNOWN_or_incomplete":sum(x["unknown_or_unfinished"] for x in traces),
       "confirmed_CCE_preferences":sum(x["confirmed_CCE_preference"] is not None for x in cells),
       "confirmed_dimensions":dict(accepted),
       "abstained_CCE_preferences":sum(x["confirmed_CCE_preference"] is None for x in cells),
       "abstention_reasons":dict(abstained),
       "ordinary_terminal_scalar_K8_non_ties":sum(x["K8_terminal_scalar_propagation"]["chosen"] is not None for x in cells),
       "ordinary_terminal_scalar_final_K_non_ties":sum(x["final_terminal_scalar_propagation"]["chosen"] is not None for x in cells),
       "terminal_scalar_not_confidence_certified":True,
       "candidate_model_forward_calls":hf.model_forwards,
       "candidate_model_score_cache_hits":hf.cached_score_lookups,
       "unique_role_observation_candidate_distributions":len(hf.cache),
       "tokenized_model_input_tokens":hf.tokenized_input_tokens,
       "actual_sampling_wall_seconds":round(time.monotonic()-start,2),
       "model_current_policy_Qpi_is_sampled_not_native_Qstar":True,
       "preregistered_protocol_sha256":sha(prereg),
       "initial_A_source_sha256":sha(args.source),
       "original_phase2_data_never_modified":True,
       "DEV_and_FINAL_never_loaded":True,
    }
    (args.output/"ADAPTIVE_CELL_DECISIONS.jsonl").write_text("".join(
        json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in cells),encoding="utf8")
    (args.output/"SUMMARY.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_SFT_FINAL_ADAPTIVE_SUMMARY",json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
