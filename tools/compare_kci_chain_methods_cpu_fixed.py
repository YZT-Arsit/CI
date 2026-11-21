r"""Small real CPU parameter-training comparison on executable KCI A->R->B->C chain.

* Shared one-parameter-set role-conditioned SharedHornActor (~4.5k params)
  makes A package choice, Coordinator B OPEN/SKIP, downstream source RAW/VERIFY,
  and BusinessExecutor COMMIT/ABORT. Forced required tool reads and legitimate
  route selection are scripted equally across methods.
* Methods: Action SFT; SFT extra same-state/counterfactual examples;
  Terminal-policy REINFORCE; CCE-reference weighted DPO surrogate;
  CCE + evidence-backed downstream DPR repair augmentation.
* Same 4 train graphs, 1 dev graph, 2 exploratory test graphs; same initial
  model seed and common 30-update Action SFT warmstart; equal 120 subsequent
  optimizer updates. DIFFERENT native evidence collection budgets reported.
* CCE uses EXACT BOUNDED NATIVE Qstar with actual verified suffix, NOT frozen
  model Qpi or real multi-agent language-model rollouts. DPR only trains from
  actual failed native policy trajectories and verifies corrective same-role
  action. No fake downstream positive labels or upstream blame.
* Not an original KCI-Align algorithm reproduction or paper-grade result.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
from collections import Counter,defaultdict
from hashlib import sha256
import json,sys,time,random
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from chain_shared_actor_cpu_fixed import (
    TASKS,SharedHornActor,environment,actor_feature,stack,
    run_episode,valid_decision_candidates,baseline_automatic_action,
    score_actions,digest,
)
from kci.diverse_ci_branch_env import FINAL
from solve_v31_longterm import native_best

DATA=ROOT/"artifacts"/"20261120_v31"/"DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01"
OUT=ROOT/"artifacts"/"20261120_v31"/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_HORN_REPAIRED_01"
SEEDS=(7,1234,2025)
WARMUP=30
FINETUNE=120
LR=0.015
BETA=3.0
METHODS=("action_sft","sft_counterfactual_coverage","terminal_reinforce",
         "cce_reference_dpo_proxy","cce_reference_dpo_plus_dpr_proxy")

def load(n):
    return [json.loads(line) for line in (DATA/n).read_text(encoding="utf-8").splitlines() if line.strip()]
ALL=load("CHAIN_CAUSAL_PAIRS_EVAL_ONLY.jsonl")
assert len(ALL)==28
GROUP={s:[w for w in ALL if TASKS[w["root"]].root_split==s]
       for s in ("train","dev","exploratory_test")}
assert [len(GROUP[s]) for s in GROUP]==[16,4,8]
assert len({r["graph"] for r in GROUP["train"]})==4
assert len({r["graph"] for r in GROUP["dev"]})==1
assert len({r["graph"] for r in GROUP["exploratory_test"]})==2

def point(observation,chosen,weight=1.0,origin="demo",rejected=None):
    options=valid_decision_candidates(observation,observation["legal_actions"])
    assert chosen in options,(chosen,options)
    if rejected is not None:assert rejected in options and rejected!=chosen
    return {"obs":deepcopy(observation),"options":options,"chosen":chosen,
            "weight":float(weight),"origin":origin,"rejected":rejected,
            "actor":observation.get("actor",observation.get("role")),
            "obs_sha256":digest(observation)}

def build_expert_dataset(rows):
    routine=[]
    counterfactual=[]
    highlevel=[]
    counts=Counter()
    for row in rows:
        env=environment(row)
        s=env.initial()
        s,a=env.transition(s,"A_READ_VERIFIED_PACKAGES")
        assert a["verdict"]=="VALID"
        obsA=env.observation(s)
        avals=row["upstream_A_values_given_exact_B_continuation"]
        if avals["PACKAGE_A"]>avals["PACKAGE_B"]:bestA="PACKAGE_A"
        elif avals["PACKAGE_B"]>avals["PACKAGE_A"]:bestA="PACKAGE_B"
        else:bestA=row["benign_package"]  # True tie: both permitted, choose benign.
        sameA="A_SEND_"+bestA
        routine.append(point(obsA,sameA,origin="expert_A_reference"))
        other="PACKAGE_A" if bestA=="PACKAGE_B" else "PACKAGE_B"
        if abs(avals[bestA]-avals[other])>1e-8:
            highlevel.append(point(obsA,sameA,
                weight=abs(avals[bestA]-avals[other]),
                origin="CCE_A_exact_native_reference",rejected="A_SEND_"+other))
        counts["expert_A_states"]+=1
        for package in ("PACKAGE_A","PACKAGE_B"):
            after,a=env.transition(s,"A_SEND_"+package)
            assert a["verdict"]=="VALID"
            for act in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                after,a=env.transition(after,act)
                assert a["verdict"]=="VALID"
            obsB=env.observation(after)
            q_open=native_best(env.native,after.native,"OPEN_PRIVATE_CHANNEL")
            q_skip=native_best(env.native,after.native,"SKIP_PRIVATE_CHANNEL")
            counts["actual_native_reference_expansions"]+=(
                q_open["states_explored"]+q_skip["states_explored"])
            counts["actual_native_reference_arms"]+=2
            b_gold=("OPEN_PRIVATE_CHANNEL" if q_open["native_exact_Qstar"]>
                    q_skip["native_exact_Qstar"] else "SKIP_PRIVATE_CHANNEL")
            b_other=("SKIP_PRIVATE_CHANNEL" if b_gold=="OPEN_PRIVATE_CHANNEL" else
                     "OPEN_PRIVATE_CHANNEL")
            b_point=point(obsB,b_gold,origin="expert_B_reference")
            counterfactual.append(b_point)
            if package==bestA:routine.append(b_point)
            if abs(q_open["native_exact_Qstar"]-q_skip["native_exact_Qstar"])>1e-8:
                highlevel.append(point(obsB,b_gold,
                    weight=abs(q_open["native_exact_Qstar"]-q_skip["native_exact_Qstar"]),
                    origin="CCE_B_exact_native_reference",rejected=b_other))
            optimal=q_open if b_gold=="OPEN_PRIVATE_CHANNEL" else q_skip
            current=after
            for i,entry in enumerate(optimal["action_trace"]):
                obs=env.observation(current)
                a=entry["action"]
                opts=valid_decision_candidates(obs,obs["legal_actions"])
                if len(opts)>1:
                    p=point(obs,a,origin="expert_native_critical_decision")
                    counterfactual.append(p)
                    if package==bestA:routine.append(p)
                    counts["expert_critical_agent_steps"]+=1
                current,audit=env.transition(current,a)
                assert audit["verdict"]==entry["actual_verdict"]
            assert current.native.done
            assert current.native.success
        counts["train_worlds_with_both_A_B_branches"]+=1
    # Crucial: EXTRA SFT sees exactly the same counterfactual B states
    # that CCE scores, not merely the "best A" branch.
    assert routine and counterfactual and highlevel
    return routine,counterfactual,highlevel,dict(counts)

def feature_cache(points):
    allf=[];lengths=[];positive=[];negative=[];sample_weight=[]
    for p in points:
        lengths.append(len(p["options"]))
        positive.append(p["options"].index(p["chosen"]))
        negative.append(p["options"].index(p["rejected"]) if p["rejected"] else -1)
        sample_weight.append(p["weight"])
        allf.extend(actor_feature(p["obs"],a) for a in p["options"])
    return {"features":stack(allf),"lengths":lengths,
            "target":positive,"negative":negative,
            "weight":sample_weight}

def supervised_loss(model,cached,kind="sft",reference=None):
    scores=model(cached["features"])
    pos=0
    losses=[]
    values=[]
    for idx,n in enumerate(cached["lengths"]):
        local=scores[pos:pos+n];pos+=n
        target=cached["target"][idx]
        if kind=="sft":
            losses.append(-torch.log_softmax(local,dim=0)[target])
        elif kind=="dpo":
            reject=cached["negative"][idx]
            if reject<0:
                losses.append(-torch.log_softmax(local,dim=0)[target])
            else:
                ref=reference[idx]
                delta=(local[target]-local[reject])-ref
                losses.append(-F.logsigmoid(BETA*delta))
        else:raise ValueError(kind)
    weights=torch.tensor(cached["weight"],dtype=torch.float32)
    if kind=="dpo":
        # Weights are grounded actual optimal value gaps. Normalize exactly
        # once on the target training split, not by dev/test decisions.
        weights=torch.clamp(weights/weights.mean(),min=.1,max=4)
    return (torch.stack(losses)*weights).sum()/weights.sum()

def ref_diff(net,cache):
    with torch.no_grad():
        flat=net(cache["features"])
        pos=0;answers=[]
        for i,n in enumerate(cache["lengths"]):
            block=flat[pos:pos+n];pos+=n
            j=cache["negative"][i]
            ans=float(block[cache["target"][i]]-block[j]) if j>=0 else 0.0
            answers.append(ans)
        return answers

def optimize_sft(model,cache,opt):
    opt.zero_grad()
    loss=supervised_loss(model,cache)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),3)
    opt.step()
    return float(loss.detach())

def optimize_dpo(model,cache,reference,opt):
    opt.zero_grad()
    loss=supervised_loss(model,cache,"dpo",reference)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),3)
    opt.step()
    return float(loss.detach())

def compute_DPR_from_failures(model,worlds,seed,max_examples=30):
    """Only actual policy failure states; never backprop on upstream legal A."""
    corrections=[];checks=Counter()
    rng=np.random.default_rng(seed)
    for world in worlds:
        env=environment(world)
        episode=run_episode(env,model,rng,greedy=False)
        checks["native_policy_rollouts"]+=1
        checks["policy_actions"]+=episode["n_policy_choices"]
        if not episode["success"]:
            checks["actual_failures"]+=1
        if episode["violation"]:checks["violations"]+=1
        decisions=episode["decision_states"]
        for item in decisions[::-1]:
            obs=item["actor_observation"]
            options=item["candidate_actions"]
            act=item["selected_action"]
            state=item["native_state_before"]
            if obs.get("current_stage")!="release":
                continue
            if act!="SEND_RAW_R" or "VERIFY_PRIVATE_R" not in options:
                continue
            bad,ab=env.native.transition(state,"SEND_RAW_R")
            good,ag=env.native.transition(state,"VERIFY_PRIVATE_R")
            if ab["verdict"]!="VIOLATION" or ag["verdict"]!="VALID":
                continue
            # True same-state corrective candidate under the same native
            # continuation search; do not claim sampled model Q-pi.
            qbad=native_best(env.native,state,"SEND_RAW_R")
            qgood=native_best(env.native,state,"VERIFY_PRIVATE_R")
            checks["DPR_reference_searches"]+=2
            checks["DPR_reference_states_visited"]+=(
                qbad["states_explored"]+qgood["states_explored"])
            if qgood["native_exact_Qstar"]<=qbad["native_exact_Qstar"]:
                continue
            p=point(obs,"VERIFY_PRIVATE_R",
                    weight=qgood["native_exact_Qstar"]-qbad["native_exact_Qstar"],
                    origin="DPR_from_ACTUAL_failed_policy_trajectory_same_role",
                    rejected="SEND_RAW_R")
            assert p["actor"]==obs.get("actor",obs.get("role"))
            assert item["selected_action"]=="SEND_RAW_R"
            assert episode["violation"]
            corrections.append(p)
            checks["verified_downstream_corrections"]+=1
            break
        if len(corrections)>=max_examples:break
    # De-duplicate exact same (state, role, action) if sampled elsewhere.
    unique={}
    for p in corrections:
        key=(p["obs_sha256"],p["chosen"],p["rejected"])
        unique.setdefault(key,p)
    checks["unique_verified_corrections"]=len(unique)
    return list(unique.values()),dict(checks)

def evaluate_model(model,rows,per_episode=False):
    outcomes=[];tot=Counter()
    for w in rows:
        env=environment(w)
        episode=run_episode(env,model,greedy=True)
        tot["episodes"]+=1
        tot["safe_success"]+=int(episode["success"])
        tot["violation"]+=int(episode["violation"])
        tot["safe_incomplete"]+=int(not episode["success"] and not episode["violation"])
        tot["invalid"]+=int(episode["invalid"])
        tot["native_steps"]+=episode.get("native_steps",episode["final"].steps)
        tot["trainable_decisions"]+=episode["n_policy_choices"]
        tot["total_return_milli"]+=round(episode["return"]*1000000)
        # Both correct A and actual downstream actions need evaluation, not
        # only success: exact native source data stored in this world.
        chosenA=next((r["action"] for r in episode["actions"] if
            r["stage"]=="A_SEND" and r["learned_decision"]),None)
        a_gold=w["upstream_A_values_given_exact_B_continuation"]
        if chosenA:
            chosen=chosenA.removeprefix("A_SEND_")
            best=max(a_gold.values())
            tot["A_best_or_tie"]+=int(abs(a_gold[chosen]-best)<1e-7)
        b_decisions=[r for r in episode["actions"] if
                     r["stage"]=="preflight" and r["learned_decision"]]
        # A/B accuracy only through actual chosen sequence; heldout oracle
        # gap precomputed from genuinely branched world is test EVAL ONLY.
        if b_decisions and chosenA:
            cname=chosenA.removeprefix("A_SEND_")
            risky=cname==w["related_package"]
            b_gold="OPEN_PRIVATE_CHANNEL" if risky else "SKIP_PRIVATE_CHANNEL"
            tot["B_correct_conditional_on_visited_history"]+=int(
                b_decisions[0]["action"]==b_gold)
        if per_episode:
            outcomes.append({
                "world":w["world"],"graph":w["graph"],
                "mode":w["source_mode"],"swap":w["swapped_package_names"],
                "native_success":episode["success"],
                "native_violation":episode["violation"],
                "native_steps":episode.get("native_steps",episode["final"].steps),
                "native_return":episode["return"],
                "actual_model_actions":[{"role":r["role"],"stage":r["stage"],
                      "action":r["action"],"verdict":r["verdict"],
                      "new_R_delivery":r["new_R_delivery"],
                      "learned":r["learned_decision"]} for r in episode["actions"]],
                "source_of_policy":"TRAINED_SHARED_CPU_ACTOR",
            })
    count=tot["episodes"]
    summary={
        "n":count,
        "safe_success":tot["safe_success"],
        "ci_violations":tot["violation"],
        "safe_incomplete":tot["safe_incomplete"],
        "invalid_actions_or_unhandled_stages":tot["invalid"],
        "mean_native_return":round(tot["total_return_milli"]/1000000/count,6),
        "mean_native_steps":round(tot["native_steps"]/count,3),
        "A_optimal_action_given_eval_truth":tot["A_best_or_tie"],
        "B_correct_on_visited_history":tot["B_correct_conditional_on_visited_history"],
        "mean_policy_decisions":round(tot["trainable_decisions"]/count,2),
    }
    return summary,outcomes

def learning_run(seed,mode,routine,expanded,high,groups):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    net=SharedHornActor()
    cached_common=feature_cache(routine)
    opt=torch.optim.Adam(net.parameters(),lr=LR)
    cum_updates=0;log=[]
    training_data_calls=Counter()
    t0=time.perf_counter()
    for i in range(WARMUP):
        loss=optimize_sft(net,cached_common,opt)
        cum_updates+=1
    ref=deepcopy(net).eval()
    for parameter in ref.parameters():parameter.requires_grad_(False)
    cache_full=feature_cache(expanded)
    cache_hi=feature_cache(high)
    reference_hi=ref_diff(ref,cache_hi)
    checkpts=(0,40,80,120)
    checkpoints=[]
    dpr_records=[]
    rng=np.random.default_rng(seed+900)
    for step in range(FINETUNE+1):
        if step in checkpts:
            dev,_=evaluate_model(net,groups["dev"])
            checkpoints.append({
                "steps_after_common_warmup":step,
                "dev":dev,
                "completed_updates":cum_updates})
        if step==FINETUNE:break
        if mode=="action_sft":
            loss=optimize_sft(net,cached_common,opt)
        elif mode=="sft_counterfactual_coverage":
            loss=optimize_sft(net,cache_full,opt)
        elif mode in ("cce_reference_dpo_proxy","cce_reference_dpo_plus_dpr_proxy"):
            opt.zero_grad()
            # DPO relative to the SAME common frozen reference. Pair is same
            # actor-visible state, same actor, native exact reference Q gap.
            L1=supervised_loss(net,cache_hi,"dpo",reference_hi)
            L2=supervised_loss(net,cached_common,"sft")
            L=L1+0.35*L2
            if mode.endswith("plus_dpr_proxy") and dpr_records:
                dpr_cache=feature_cache(dpr_records)
                dpr_ref=ref_diff(ref,dpr_cache)
                L3=supervised_loss(net,dpr_cache,"dpo",dpr_ref)
                L=L+0.55*L3
            L.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3)
            opt.step()
            loss=float(L.detach())
        elif mode=="terminal_reinforce":
            # A terminal-only on-policy REINFORCE update: sample same train
            # 16 worlds with policy probabilities at ALL learned decisions,
            # observe solely each sampled trajectory's actual native result.
            rollout=[]
            for world in groups["train"]:
                env=environment(world)
                ep=run_episode(env,net,rng,greedy=False,collect_logprob=True)
                rollout.append(ep)
                training_data_calls["terminal_native_episodes"]+=1
                training_data_calls["terminal_native_actor_steps"]+=len(ep["actions"])
            returns=np.array([x["return"] for x in rollout],np.float32)
            baseline=float(returns.mean())
            opt.zero_grad()
            terms=[]
            for ep in rollout:
                if ep["logp"]:
                    terms.append(
                        -(float(ep["return"])-baseline)*torch.stack(ep["logp"]).sum())
            if terms:
                loss_tensor=torch.stack(terms).mean()
                # No additional oracle or step-level preferred labels.
                loss_tensor.backward()
                nn.utils.clip_grad_norm_(net.parameters(),3)
                opt.step()
                loss=float(loss_tensor.detach())
            else:loss=0.0
        else:raise ValueError(mode)
        cum_updates+=1
        if mode=="cce_reference_dpo_plus_dpr_proxy" and step in (9,29,49,69,89,109):
            recovered,counts=compute_DPR_from_failures(
                net,groups["train"],seed+step+123)
            training_data_calls.update(counts)
            for rec in recovered:
                if rec["obs_sha256"] not in {r["obs_sha256"] for r in dpr_records}:
                    dpr_records.append(rec)
            training_data_calls["DPR_event_unique_corrections"]=len(dpr_records)
        if step in (0,39,79,119):
            log.append({"finetune_step":step+1,"loss":round(loss,6)})
    scores={name:evaluate_model(net,rows)[0] for name,rows in groups.items()}
    full={name:evaluate_model(net,rows,per_episode=True)[1]
          for name,rows in groups.items() if name!="train"}
    # Source-vs-downstream accountability: no DPR record should contain A.
    assert all(p["actor"]!="RecordsAgent" for p in dpr_records)
    return {
        "seed":seed,"mode":mode,"scores":scores,
        "checkpoints":checkpoints,
        "training_loss_samples":log,
        "training_native_calls":dict(training_data_calls),
        "DPR_actual_failure_verified_unique_pairs":len(dpr_records),
        "DPR_roles":dict(Counter(p["actor"] for p in dpr_records)),
        "initial_common_SFT_updates":WARMUP,
        "subsequent_optimizer_updates":FINETUNE,
        "train_gold_oracle_action_states":len(routine),
        "train_counterfactual_states":len(expanded),
        "CCE_exact_native_action_comparison_states":len(high),
        "model_params":sum(p.numel() for p in net.parameters()),
        "training_wall_seconds":round(time.perf_counter()-t0,3),
        "dev_test_episode_traces":full,
    }

def main():
    if OUT.exists():raise FileExistsError(OUT)
    common,cf,cce,counts=build_expert_dataset(GROUP["train"])
    print("DATA CONTRACT",json.dumps({
        "train_roots":len(GROUP["train"]),"dev":len(GROUP["dev"]),
        "exploratory_test":len(GROUP["exploratory_test"]),
        "common_expert_states":len(common),
        "counterfactual_state_labeled_actions":len(cf),
        "reference_CCE_A_B_comparisons":len(cce),
        "real_native_reference":counts},ensure_ascii=False),flush=True)
    tests=[]
    for mode in METHODS:
        for seed in SEEDS:
            result=learning_run(seed,mode,common,cf,cce,GROUP)
            tests.append(result)
            print("COMPARISON",mode,seed,
                  "dev",result["scores"]["dev"],
                  "test",result["scores"]["exploratory_test"],
                  "DPR_pairs",result["DPR_actual_failure_verified_unique_pairs"],
                  "time",result["training_wall_seconds"],flush=True)
    aggregate={}
    for method in METHODS:
        aggregate[method]={}
        for split in ("train","dev","exploratory_test"):
            x=[r["scores"][split] for r in tests if r["mode"]==method]
            aggregate[method][split]={
                key:{
                    "mean":round(float(np.mean([v[key] for v in x])),6),
                    "sample_SD":round(float(np.std([v[key] for v in x],ddof=1)),6),
                    "per_seed":[v[key] for v in x]}
                for key in ("safe_success","ci_violations","safe_incomplete",
                            "mean_native_return","mean_native_steps",
                            "A_optimal_action_given_eval_truth",
                            "B_correct_on_visited_history")}
        aggregate[method]["DPR_unique_verified_pairs"]=[
            r["DPR_actual_failure_verified_unique_pairs"]
            for r in tests if r["mode"]==method]
    report={
        "status":"CPU_SHARED_ROLE_POLICY_METHOD_COMPARISON_REAL_NATIVE_ROLLOUTS",
        "model":"ONE shared role-conditioned CPU SharedHornActor, zero-preserving exact Horn AND/OR on trusted 0/1 observations (4.5k learned params), across RecordsAgent, Coordinator, source role, BusinessExecutor",
        "key_limit":"Repaired rule-inference feature encoder now equals the explicit source-visible Horn closure; this STRONG engineered reasoning prior may trivialize knowledge inference. No trained LLM, not full original CCE/DPR/GRPO.",
        "dataset":"Original 7-graph dynamic A->R->B->C native pilot; all files frozen",
        "train_roots":len(GROUP["train"]),
        "dev_roots":len(GROUP["dev"]),
        "exploratory_test_roots":len(GROUP["exploratory_test"]),
        "split_unit":"entire Horn rule family, not synthetic world variant",
        "training_protocol":{
            "seeds":SEEDS,"shared_warmstart_SFT_updates":WARMUP,
            "matched_additional_optimizer_updates":FINETUNE,
            "exact_same_Adam_lr":LR,"DPO_beta":BETA,
            "fixed_business_tools":"same predesigned native simulator across all methods",
            "forced_tool_steps":"A source READ, inspect business, selected legitimate business route, required source READ order, signed attest",
            "shared_model_controls":"A_SEND package; B OPEN/SKIP/ABORT; source RAW/VERIFY/ABORT; C COMMIT/ABORT",
            "ActionSFT":"One native expert trajectory per training world",
            "SFT_counterfactual_coverage":"Same counterfactual B and source actor-visible states as CCE; use only optimal action labels with no value magnitudes",
            "CCE_reference_dpo_proxy":"Same-state pairwise native-exact Qstar weighted DPO surrogate relative to shared frozen SFT reference, plus same common source imitation",
            "CCE_DPR_proxy":"CCE plus targeted same-role corrections collected from ACTUAL failed sampled CPU model trajectories; extra rollout/search budget logged",
            "terminal_reinforce":"On-policy sampled complete native episode reward for all model decisions, REINFORCE with batch baseline; DOES NOT see unselected arm return",
            "DPR_eligibility":"Actually sampled source RAW causes incrementally protected knowledge; VERIFY must be legal and exact native Qstar-better. No upstream A blame.",
        },
        "evidence_generation":counts,
        "training_points":{"expert":len(common),"counterfactual_coverage":len(cf),
                           "CCE_exact_native_A_B":len(cce)},
        "aggregate":aggregate,
        "per_seed_and_heldout_full_native_model_action_paths":tests,
        "all_norms_and_business_fixtures_synthetic":True,
        "heldout_graphs_exploratory_not_blind":True,
        "no_Qwen_weight_updates":True,
        "no_paid_API_or_GPU":True,
    }
    OUT.mkdir(parents=True)
    (OUT/"RESULTS.json").write_text(json.dumps(
        report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY",json.dumps({"status":report["status"],
        "train_samples":report["training_points"],
        "aggregate":aggregate},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
