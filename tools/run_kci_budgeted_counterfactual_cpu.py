"""Budgeted counterfactual rollouts for scaling KCI from exact CPU Qpi to LLM sampling.

All data come from existing REAL native A->R->B->C histories. Simulate a
frozen learned agent's stochastic downstream action distribution with state-
matched counterfactual forks. No oracle Qstar or Qpi is used for the teacher;
exact Qpi is separately computed ONLY to validate sampling diagnostics.

Paired/common-random-number (CRN) rollouts use the same RNG stream after
the two *forced* legal A actions. Both marginals are genuine samples under
the same frozen policy. Identical samples are NOT independent replicates.
"""
from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import json
import math
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"), str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import (
    SharedHornActor,score_actions,valid_decision_candidates,baseline_automatic_action,
)
from kci_exact_policy_continuation_cpu import (
    environment_public,terminal_vector,FrozenPolicyValue
)
from kci.diverse_ci_branch_env import digest
from run_kci_downstream_residual_cpu import RepairedDownstream
from run_kci_role_local_credit_cpu import load_base

torch.set_num_threads(2)
SEEDS=(7,1234,2025)
REPS=3
KS=(2,4,8,16)
MAX_K=max(KS)
GAP=0.010
CONSENSUS=.75
OUT=ROOT/"artifacts"/"20261121_kci_budgeted_counterfactual_cpu_v2"


class FrozenNativeSampler:
    def __init__(self,env,policy):
        self.env=env
        self.policy=policy.eval()
        self.options_cache={}
        self.state_visits=0
        self.policy_queries=0
        self.episodes=0
        self.native_steps=0

    def draw(self,state,seed):
        rng=np.random.default_rng(seed)
        s=state
        for step in range(70):
            self.state_visits+=1
            if s.native.done:
                self.episodes+=1
                return terminal_vector(self.env,s)
            key=digest(asdict(s))
            if key not in self.options_cache:
                obs=self.env.observation(s)
                opts=list(self.env.options(s))
                dec=valid_decision_candidates(obs,opts)
                if len(dec)>=2:
                    with torch.no_grad():
                        logits=score_actions(self.policy,obs,dec)
                        probabilities=torch.softmax(logits,dim=0).cpu().numpy().tolist()
                    entry=(dec,probabilities)
                    self.policy_queries+=1
                else:
                    action=baseline_automatic_action(self.env,s)
                    if action is None:
                        raise AssertionError(("Missing required native adapter action",obs))
                    entry=([action],[1.0])
                self.options_cache[key]=entry
            actions,probs=self.options_cache[key]
            if len(actions)==1:
                action=actions[0]
            else:
                u=rng.random()
                cum=0.
                action=actions[-1]
                for a,p in zip(actions,probs):
                    cum+=p
                    if u<cum:
                        action=a
                        break
            nxt,audit=self.env.transition(s,action)
            self.native_steps+=1
            if audit["verdict"] not in ("VALID","VIOLATION"):
                raise AssertionError((audit,action))
            s=nxt
        raise AssertionError("Native episode exceeded fixed safety horizon")


def load_policy(seed,version):
    base,_=load_base(seed)
    if version=="initial":
        return base.eval()
    if version=="downstream_repaired":
        policy=RepairedDownstream(base).eval()
        file=(ROOT/"artifacts"/"20261121_kci_downstream_residual_cpu_v1" /
              "CHECKPOINTS"/("A_Qpi_after_downstream_seed"+str(seed)+".pt"))
        saved=torch.load(file,map_location="cpu",weights_only=True)
        policy.B.load_state_dict(saved["B"])
        policy.R.load_state_dict(saved["R"])
        return policy.eval()
    raise ValueError(version)


def fork_setup(row):
    env=environment_public(row)
    st,audit=env.transition(env.initial(),"A_READ_VERIFIED_PACKAGES")
    assert audit["verdict"]=="VALID"
    options=("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")
    children=[]
    for action in options:
        n,v=env.transition(st,action)
        assert v["verdict"]=="VALID"
        children.append(n)
    return env,st,children


def sample_pair_stream(env,children,sampler,k,seed,paired):
    outputs=[]
    for i in range(k):
        entropy=seed+104729*i
        seed1=entropy if paired else entropy+1000030007
        a=sampler.draw(children[0],entropy)
        b=sampler.draw(children[1],seed1)
        outputs.append((a,b))
    return outputs


def classify(pairs,k,gap=GAP,consensus=CONSENSUS):
    x=pairs[:k]
    deltas=[a[0]-b[0] for a,b in x]
    pos=sum(d>1e-8 for d in deltas)
    neg=sum(d< -1e-8 for d in deltas)
    sign=1 if sum(deltas)>0 else -1 if sum(deltas)<0 else 0
    agree=max(pos,neg)/k
    mean=float(np.mean(deltas))
    accepted=(abs(mean)>=gap and agree>=consensus and
              sign==(1 if pos>neg else -1 if neg>pos else 0))
    return {"mean_gap":mean,"direction":(sign if accepted else 0),
            "raw_direction":sign,"agreement":agree,"accepted":bool(accepted),
            "positive_pairs":pos,"negative_pairs":neg,
            "sample_mean_A":float(np.mean([a[0] for a,b in x])),
            "sample_mean_B":float(np.mean([b[0] for a,b in x]))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    results=[]
    totals=Counter()
    start=time.time()
    for version in ("initial","downstream_repaired"):
        for seed in SEEDS:
            policy=load_policy(seed,version)
            model_hash=sha256(str(tuple(
                (k,float(v.detach().reshape(-1)[0]))
                for k,v in policy.state_dict().items())).encode()).hexdigest()
            for split in ("train","dev","exploratory_test"):
                for row in GROUP[split]:
                    env,origin,children=fork_setup(row)
                    sampler=FrozenNativeSampler(env,policy)
                    evalr=FrozenPolicyValue(env,policy)
                    exact=[evalr.q(origin,a) for a in
                           ("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B")]
                    true_gap=exact[0][0]-exact[1][0]
                    qstar=row["upstream_A_values_given_exact_B_continuation"]
                    ideal_gap=qstar["PACKAGE_A"]-qstar["PACKAGE_B"]
                    for coupled in (True,False):
                        for rep in range(REPS):
                            rootseed=(seed*1000003+
                                      int(sha256((row["world"]+version).encode()).hexdigest()[:8],16)+
                                      rep*8191)
                            draws=sample_pair_stream(
                                env,children,sampler,MAX_K,rootseed,coupled)
                            for k in KS:
                                verdict=classify(draws,k)
                                correct=(int(math.copysign(1,verdict["direction"])) ==
                                         int(math.copysign(1,true_gap))
                                         if verdict["direction"] and abs(true_gap)>1e-8 else None)
                                results.append({
                                    "version":version,"seed":seed,"split":split,
                                    "world":row["world"],"source_mode":row["source_mode"],
                                    "paired":coupled,"replicate":rep,"K":k,
                                    "exact_Qpi_gap_A_minus_B_EVAL_ONLY":true_gap,
                                    "exact_Qstar_gap_A_minus_B_EVAL_ONLY":ideal_gap,
                                    **verdict,
                                    "wrong_accepted":bool(correct is False),
                                    "right_accepted":bool(correct is True),
                                    "strict_Qpi":abs(true_gap)>1e-8,
                                    "strict_Qstar":abs(ideal_gap)>1e-8,
                                })
                    totals["native_steps"]+=sampler.native_steps
                    totals["episodes"]+=sampler.episodes
                    totals["sampled_unique_policy_queries"]+=sampler.policy_queries
                    totals["exact_eval_states"]+=evalr.queries
            print("SAMPLED",version,seed,"elapsed",round(time.time()-start,2),
                  "episodes",totals["episodes"],flush=True)
    summary={}
    for version in ("initial","downstream_repaired"):
        summary[version]={}
        for split in ("train","dev","exploratory_test"):
            summary[version][split]={}
            for paired in (True,False):
                typ="paired_CRN" if paired else "independent"
                summary[version][split][typ]={}
                for k in KS:
                    xs=[x for x in results if x["version"]==version and
                        x["split"]==split and x["paired"]==paired and x["K"]==k
                        and x["strict_Qpi"]]
                    accepted=sum(x["accepted"] for x in xs)
                    mistakes=sum(x["wrong_accepted"] for x in xs)
                    right=sum(x["right_accepted"] for x in xs)
                    considered=len(xs)
                    raw_error=sum(x["raw_direction"]*
                                  x["exact_Qpi_gap_A_minus_B_EVAL_ONLY"]<0
                                  for x in xs)
                    summary[version][split][typ][str(k)]={
                        "strict_pairs_over_seeds_replicates":considered,
                        "accepted":accepted,"right_accepted":right,
                        "wrong_accepted":mistakes,
                        "acceptance_rate":accepted/considered if considered else None,
                        "wrong_rate_over_accepted":mistakes/accepted if accepted else None,
                        "raw_sign_error_rate":raw_error/considered if considered else None,
                        "continuation_episodes_per_two_action_decision":2*k
                    }
    payload={
        "status":"CPU_SAMPLE_BASED_POLICY_CONDITIONED_CREDIT_GATE",
        "all_source_norms_synthetic":True,
        "model":"original shared CPU Horn actor and downstream repaired actor",
        "frozen_model_policy_sampling":True,
        "exact_Qpi_only_for_separate_audit":True,
        "Qstar_never_used_to_accept_train_CCE_preferences":True,
        "no_test_information_in_train_teacher":True,
        "paired_CRN_is_variance_reduction_not_independent_samples":True,
        "K_values":KS,"replicate_count":REPS,
        "gap_threshold":GAP,"paired_sign_consensus":CONSENSUS,
        "not_blind_task_family":True,"not_LLM_or_GPU":True,
        "resource_counts":dict(totals),"summary":summary,
        "full_records":results
    }
    (OUT/"RESULTS.json").write_text(
        json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("RESULT_SUMMARY",json.dumps(summary["downstream_repaired"]["train"],
                                     ensure_ascii=False),flush=True)
    print("RESULT_FILE",OUT/"RESULTS.json",flush=True)

if __name__=="__main__":
    main()
