r"""Re-estimate exact frozen Qpi before/after EACH real CPU policy update.

NO stale teacher: all 4 × 3 post-training models are reloaded and re-evaluated
as the current policy. All 28 ORIGINAL source worlds are evaluated, including
train/dev/exploratory roots (no new scenarios). Full unforced policy Qpi
includes A's abort option; same-state forced A action Qpi includes policy
downstream action distributions and actual native transitions.

Qstar is referenced only to score offline, NEVER in actor input.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import json,time,sys,torch,math
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_exact_frozen_qpi_cpu import (
 GROUP,ROWS,SEEDS,load_frozen_policy,SharedHornActor,
 ExactQpi,prepare_A,environment,qstar_A_from_original)
from chain_shared_actor_cpu_action_interaction import (
 score_actions,valid_decision_candidates,digest)
from compare_kci_chain_methods_cpu_action_interaction import evaluate_model
BASE=ROOT/"artifacts"/"20261121_policy_gap_pilot_01"
TRAIN=BASE/"FOUR_WAY_CPU_TRAINED_POLICY_RESULTS_03"
OUT=BASE/"EXACT_QPI_BEFORE_AFTER_FOUR_METHODS_02"
if OUT.exists():raise FileExistsError(OUT)
assert (TRAIN/"TRAINING_SUMMARY.json").is_file()
meta=json.loads((TRAIN/"TRAINING_SUMMARY.json").read_text(encoding="utf-8"))
assert meta["status"]=="FOUR_WAY_SHARED_CPU_AGENT_REAL_NATIVE_TRAINING_COMPLETE"
methods=meta["methods"]
assert len(methods)==4
OUT.mkdir(parents=True)
run_summary=[]
with (OUT/"ALL_EXACT_PRE_POST_POLICY_REEVALUATIONS.jsonl").open(
      "x",encoding="utf-8",buffering=1) as stream:
  for seed in SEEDS:
    base_model,base_meta=load_frozen_policy("action_sft",seed)
    for method in methods:
      checkpoint=TRAIN/"MODEL_WEIGHTS"/f"{method}_seed{seed}.pt"
      data=torch.load(checkpoint,map_location="cpu",weights_only=True)
      assert data["seed"]==seed and data["method"]==method
      assert data["frozen_SFT_start_checkpoint_sha256"]==base_meta["checkpoint_sha256"]
      assert data["source_dataset_sha256"]==base_meta["source_data_sha256"]
      post_model=SharedHornActor()
      post_model.load_state_dict(data["model_state_dict"])
      post_model.eval()
      for p in post_model.parameters():p.requires_grad_(False)
      post_sha=sha256(checkpoint.read_bytes()).hexdigest()
      counts=Counter()
      mean=defaultdict(float)
      start=time.perf_counter()
      for epoch_tag,model,policy_sha in (
        ("before",base_model,base_meta["checkpoint_sha256"]),
        ("after",post_model,post_sha)):
        for split in ("train","dev","exploratory_test"):
          for row in GROUP[split]:
            env=environment(row)
            value=ExactQpi(env,model,policy_sha)
            source=prepare_A(env)
            forced=value.evaluate_A_arms(source)
            full=value.walk(env.initial())
            assert abs(sum(full[1:4])-1)<1e-7
            # Verify native action probability at the SAME signed A observation:
            obs=env.observation(source)
            choices=valid_decision_candidates(obs,env.options(source))
            assert set(choices)=={"A_SEND_PACKAGE_A","A_SEND_PACKAGE_B","ABORT_CHAIN"}
            pp=value.probs(obs,choices)
            d={k:v for k,v in zip(choices,pp)}
            conditional_expected=(d["A_SEND_PACKAGE_A"]*forced["arms"]["PACKAGE_A"]["Qpi"]+
                d["A_SEND_PACKAGE_B"]*forced["arms"]["PACKAGE_B"]["Qpi"])
            # ABORT_CHAIN returns 0; READ is a fixed adapter action before A.
            assert abs(conditional_expected-full[0])<2e-6,(row["world"],method,seed,conditional_expected,full[0])
            star=qstar_A_from_original(row,verify=False)
            star_gap=round(star["PACKAGE_A"]-star["PACKAGE_B"],9)
            pi_gap=forced["Qpi_A_minus_B"]
            strict=abs(star_gap)>1e-8
            conflict=bool(strict and star_gap*pi_gap< -1e-9)
            # Genuine B-preflight action probabilities under each ACTUAL
            # previously delivered A source packet. Eval-only public risk
            # labels are not used in training or the downstream actor.
            B={}
            for pkg in ("PACKAGE_A","PACKAGE_B"):
              st,rep=env.transition(source,"A_SEND_"+pkg)
              assert rep["verdict"]=="VALID"
              for a in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                  st,rep=env.transition(st,a)
                  assert rep["verdict"]=="VALID"
              bo=env.observation(st)
              opts=valid_decision_candidates(bo,env.options(st))
              assert "OPEN_PRIVATE_CHANNEL" in opts and "SKIP_PRIVATE_CHANNEL" in opts
              vp=value.probs(bo,opts)
              legal=dict(zip(opts,vp))
              risky=pkg==row["related_package"]
              best="OPEN_PRIVATE_CHANNEL" if risky else "SKIP_PRIVATE_CHANNEL"
              B[pkg]={
                 "native_actual_receiver_ledger_sha256":digest(st.native.ledger),
                 "B_actor_observation_sha256":digest(bo),
                 "B_role":bo["actor"],"actual_source_package_risky_for_future":risky,
                 "prob_correct_B":legal[best],
                 "prob_OPEN":legal["OPEN_PRIVATE_CHANNEL"],
                 "prob_SKIP":legal["SKIP_PRIVATE_CHANNEL"],
                 "most_likely_B_action":max(legal,key=legal.get),
              }
            bflip=bool(B[row["related_package"]]["most_likely_B_action"]==
                        "OPEN_PRIVATE_CHANNEL" and
                        B[row["benign_package"]]["most_likely_B_action"]=="SKIP_PRIVATE_CHANNEL")
            item={
              "policy_kind":method,"seed":seed,"policy_state":epoch_tag,
              "checkpoint_sha256":policy_sha,"world":row["world"],
              "split":split,"graph_family":row["graph"],
              "Qstar_A_minus_B_EVAL_ONLY":star_gap,
              "exact_Qpi_A_minus_B":pi_gap,
              "Qstar_vs_current_Qpi_strict_conflict":conflict,
              "Qstar_true_tie":not strict,
              "full_current_policy_native_Qpi":round(full[0],9),
              "full_current_policy_success_probability":round(full[1],9),
              "full_current_policy_leak_probability":round(full[2],9),
              "full_current_policy_incomplete_probability":round(full[3],9),
              "full_current_policy_expected_steps":round(full[4],7),
              "full_policy_A_action_probabilities":d,
              "B_after_each_real_A_delivery":B,
              "B_pairwise_knowledge_flip_correct":bflip,
              "forced_A_native_Qpi_and_full_witness":forced,
              "native_cost":{
                 "states":value.count["native_states"],
                 "edges":value.count["actual_native_transition_edges"],
                 "policy_softmax_calls":value.count["unique_frozen_actor_softmax_calls"],
                 "full_native_enumeration":True,
                 "edge_evidence_fingerprint":value.edge_fingerprint.hexdigest(),
              },
              "model_Qpi_recomputed_not_reused_from_initial_teacher":True,
            }
            stream.write(json.dumps(item,ensure_ascii=False,separators=(",",":"),sort_keys=True)+"\n")
            key=epoch_tag+"_"+split
            counts[key+"_worlds"]+=1
            counts[key+"_strict_Qstar_Qpi_conflicts"]+=int(conflict)
            counts[key+"_Qstar_ties"]+=int(not strict)
            counts[key+"_B_flip_correct"]+=int(bflip)
            counts[key+"_native_states"]+=value.count["native_states"]
            counts[key+"_actor_calls"]+=value.count["unique_frozen_actor_softmax_calls"]
            mean[key+"_full_return_sum"]+=full[0]
            mean[key+"_leak_prob_sum"]+=full[2]
      for phase in ("before","after"):
        for split in ("train","dev","exploratory_test"):
          key=phase+"_"+split
          mean[key+"_mean_Qpi_full_return"]=round(mean[key+"_full_return_sum"]/counts[key+"_worlds"],7)
          mean[key+"_mean_CI_violation_probability"]=round(
              mean[key+"_leak_prob_sum"]/counts[key+"_worlds"],7)
      results={
         "seed":seed,"method":method,
         "initial_SFT_sha256":base_meta["checkpoint_sha256"],
         "trained_model_sha256":post_sha,
         "elapsed_sec":round(time.perf_counter()-start,3),
         "counts":dict(counts),
         "means":{k:v for k,v in mean.items() if "mean" in k},
      }
      run_summary.append(results)
      print("PREPOST",method,seed,
        "TEST_CONFLICT",counts["before_exploratory_test_strict_Qstar_Qpi_conflicts"],
             "TO",counts["after_exploratory_test_strict_Qstar_Qpi_conflicts"],
        "TEST_MEAN_RETURN",round(mean["before_exploratory_test_mean_Qpi_full_return"],5),
            "TO",round(mean["after_exploratory_test_mean_Qpi_full_return"],5),
        "TEST_LEAK_PROB",round(mean["before_exploratory_test_mean_CI_violation_probability"],5),
            "TO",round(mean["after_exploratory_test_mean_CI_violation_probability"],5),
        flush=True)
assert len(run_summary)==12
report={
 "status":"ALL_FOUR_METHODS_EXACT_QPI_RECOMPUTED_BEFORE_AND_AFTER_NATIVE",
 "original_worlds_per_method_seed":28,
 "models":len(run_summary),
 "frozen_sft_initial_checkpoint_used_for_each_method":True,
 "all_post_checkpoints_reloaded_not_cached_policy_values":True,
 "Qstar_labels_only_for_posthoc_conflict_scoring":True,
 "all_native_business_and_recipient_ledger_checks_enabled":True,
 "no_dev_or_test_labels_in_training":True,
 "not_language_model_training":True,
 "runs":run_summary,
 "source_raw_evidence_sha256":sha256(
     (OUT/"ALL_EXACT_PRE_POST_POLICY_REEVALUATIONS.jsonl").read_bytes()).hexdigest(),
}
(OUT/"SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("FINAL_PREPOST",json.dumps({"status":report["status"],"models":len(run_summary),
      "evidence_sha256":report["source_raw_evidence_sha256"]},ensure_ascii=False),flush=True)
