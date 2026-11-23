"""Pre-registered CCE after one-shot shared-team Action SFT.

Two arms, each from SAME native A_SEND state; immediate current-CI is VALID.
K8 = exploration and direct pre/post comparison with old seeds.
Adaptive K16/K32 confirmation uses NEW disjoint seed blocks.
No posthoc threshold relaxation; CI-first, task success, safe-success cost.
Familywise Bonferroni over 14 states x two confirmation looks, exact TWO-SIDED
paired sign tests; per-branch empirical Wilson 95% intervals reported.
Even accepted labels estimate only the specific frozen team policy, not Qstar.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from math import comb,sqrt
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"artifacts/20261123_cce_phase2_mechanism_v1"
OUT=ROOT/"artifacts/20261124_cce_phase2_post_sft_v1"
FIRST_SEEDS=(7,1234,2025,42,314,2718,1618,99)
BUDGETS=(8,16,32)
MAX_SEED_PER_ACTION=32
N_CELLS=14
ALPHA_FWER=0.05
ALPHA_LOCAL_SCREEN=0.05
# K16 and K32 are the two confirmatory looks. K8 is screening only.
ALPHA_CONFIRM=ALPHA_FWER/(N_CELLS*2)
MIN_DISCORDANT=5
MIN_BOTH_SAFE_FOR_COST=8
ACTION_A="A_SEND_PACKAGE_A";ACTION_B="A_SEND_PACKAGE_B"
ACTIONS=(ACTION_A,ACTION_B)
def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def seeds32():
    z=list(FIRST_SEEDS)
    for n in range(24):
        seed=int.from_bytes(sha256(f"KCI-CCE-20261124-INDEPENDENT-SEED-{n}".encode()).digest()[:4],"little") & 0x7fffffff
        assert seed not in z
        z.append(seed)
    assert len(z)==len(set(z))==32
    return tuple(z)
SEEDS32=seeds32()

def paired_sign(a,b):
    assert len(a)==len(b) and len(a)>=1
    plus=sum(x>y for x,y in zip(a,b))
    minus=sum(x<y for x,y in zip(a,b))
    disc=plus+minus
    p=min(1.,2*sum(comb(disc,j) for j in range(max(plus,minus),disc+1))/(2**disc)) if disc else 1.
    return {"n":len(a),"n_discordant":disc,"A_better":plus,"B_better":minus,
       "ties":len(a)-disc,"exact_two_sided_p":p,
       "empirical_sign":"A" if plus>minus else "B" if minus>plus else "TIE"}

def wilson95(num,den):
    assert den>0 and 0<=num<=den
    z=1.95996398454005
    phat=num/den;factor=1+z*z/den
    mean=(phat+z*z/(2*den))/factor
    half=z*sqrt(phat*(1-phat)/den+z*z/(4*den*den))/factor
    return [max(0.,mean-half),min(1.,mean+half)]

def terminal_reward(row):
    if row["ci_violation"]:return -1.
    if row["ci_safe_success"]:return 1.
    return 0.

def terminal_propagation(a,b):
    assert len(a)==len(b)>0
    av=sum(map(terminal_reward,a))/len(a)
    bv=sum(map(terminal_reward,b))/len(b)
    # fixed full-sample point estimate: no confidence or cost gate, so not a
    # statistically certified teacher
    return {"A_mean":av,"B_mean":bv,"difference":av-bv,
      "chosen":ACTION_A if av>bv else ACTION_B if bv>av else None,
      "confidence_certified":False,
      "reward":"CI-safe task+1, actual CI violation-1, abort/deadline0"}

def stage_signal(a,b,*,alpha,allow_confirm,seed_block_check):
    assert len(a)==len(b)
    n=len(a)
    assert [x["seed"] for x in a]==[x["seed"] for x in b]==list(SEEDS32[:n])
    assert all(x["A_immediate_verdict"]=="VALID" and not x["A_immediate_CI_violation"] for x in a+b)
    summary={
      "decision":None,"status":None,"dimension":None,
      "confirmed":False,"alpha":alpha,
      "paired_sign":None,"block_consistency":None,
      "risk_bounds":{},"outcomes":{}}
    for act,rows in ((ACTION_A,a),(ACTION_B,b)):
        summary["outcomes"][act]={
          "CI_violations":sum(x["ci_violation"] for x in rows),
          "CI_safe_success":sum(x["ci_safe_success"] for x in rows),
          "safe_abort":sum(x["aborted"] and not x["ci_violation"] for x in rows),
          "deadline_missed":sum(x["deadline_missed"] for x in rows),
          "UNKNOWN_or_unfinished":sum(x["unknown_or_unfinished"] for x in rows),
          "safe_success_costs":[x["native_total_steps"] for x in rows if x["ci_safe_success"]]}
        summary["risk_bounds"][act]=wilson95(summary["outcomes"][act]["CI_violations"],n)
    if any(x["unknown_or_unfinished"] for x in a+b):
        summary["status"]="ABSTAIN_UNKNOWN";return summary
    # Strict priority. An observed risk contrast that is not statistically
    # resolved may NEVER be outweighed by downstream success/cost differences.
    va=[int(not x["ci_violation"]) for x in a]
    vb=[int(not x["ci_violation"]) for x in b]
    risk=paired_sign(va,vb)
    if risk["n_discordant"]>0:
        dimension="CI_safety";test=risk
        x,y=va,vb
    else:
        sa=[int(x["ci_safe_success"]) for x in a]
        sb=[int(x["ci_safe_success"]) for x in b]
        success=paired_sign(sa,sb)
        if success["n_discordant"]>0:
            dimension="CI_safe_completion";test=success
            x,y=sa,sb
        else:
            index=[j for j in range(n) if a[j]["ci_safe_success"] and b[j]["ci_safe_success"]]
            summary["joint_safe_cost_sample_count"]=len(index)
            if len(index)<MIN_BOTH_SAFE_FOR_COST:
                summary["status"]="ABSTAIN_INSUFFICIENT_PAIRED_SAFE_SUCCESS_FOR_COST"
                return summary
            dimension="cost_given_both_safe_success"
            x=[-a[j]["native_total_steps"] for j in index]
            y=[-b[j]["native_total_steps"] for j in index]
            test=paired_sign(x,y)
            if test["n_discordant"]==0:
                summary["status"]="ABSTAIN_EQUAL_COST"
                summary["dimension"]=dimension
                summary["paired_sign"]=test
                return summary
    summary["dimension"]=dimension
    summary["paired_sign"]=test
    if test["n_discordant"]<MIN_DISCORDANT:
        summary["status"]="ABSTAIN_FEW_DISCORDANT_TRAJECTORIES"
        return summary
    if test["exact_two_sided_p"]>alpha:
        summary["status"]="ABSTAIN_UNCERTAIN_"+dimension
        return summary
    # Seed dependence control: the hypothesized direction must appear within
    # both independent seed blocks. This is a necessary, not sufficient check.
    if seed_block_check:
        if dimension=="cost_given_both_safe_success":
            # x,y are successful-only and cannot be split by array position.
            block_data=[]
            for sl in (range(0,n//2),range(n//2,n)):
                ids=[j for j in sl if a[j]["ci_safe_success"] and b[j]["ci_safe_success"]]
                block_data.append(paired_sign([-a[j]["native_total_steps"] for j in ids],
                                [-b[j]["native_total_steps"] for j in ids]) if ids else None)
        else:
            block_data=[paired_sign(x[:n//2],y[:n//2]),paired_sign(x[n//2:],y[n//2:])]
        expected=test["empirical_sign"]
        consistent=(all(r is not None and
                      (r["A_better"]>0 if expected=="A" else r["B_better"]>0) and
                      (r["A_better"]>=r["B_better"] if expected=="A" else r["B_better"]>=r["A_better"])
                      for r in block_data))
        summary["block_consistency"]={"passed":consistent,"blocks":block_data}
        if not consistent:
            summary["status"]="ABSTAIN_SEED_BLOCK_INCONSISTENCY"
            return summary
    summary["decision"]=ACTION_A if test["empirical_sign"]=="A" else ACTION_B
    summary["status"]="ACCEPT_"+dimension
    summary["confirmed"]=allow_confirm
    return summary

def summarize_pair(a,b,n):
    assert len(a)==len(b)==n
    k8=stage_signal(a[:8],b[:8],alpha=ALPHA_LOCAL_SCREEN,
                    allow_confirm=False,seed_block_check=False)
    # In this protocol K8 is observational/screening, never a training label
    # regardless of the nominal within-case exact p value.
    k8["confirmed"]=False
    k8["K8_is_exploratory_only"]=True
    full=stage_signal(a,b,alpha=ALPHA_CONFIRM if n in (16,32) else ALPHA_LOCAL_SCREEN,
                      allow_confirm=n in (16,32),seed_block_check=n>=16)
    if n==8:full["confirmed"]=False
    return {"K8_screen":k8,"current_look":full,"terminal_K8":terminal_propagation(a[:8],b[:8]),
        "terminal_current_n":terminal_propagation(a,b)}

def should_continue(last,n):
    assert n in BUDGETS
    if n==32:return False,"MAX_K32"
    if n==8 and all(x["CI_violations"]==x["CI_safe_success"]==x["deadline_missed"]==0
                    for x in last["outcomes"].values()):
        return False,"FUTILITY_K8_NO_VIOLATION_SUCCESS_OR_DEADLINE_IN_EITHER_BRANCH"
    if n>=16 and last["confirmed"]:
        return False,"CONFIRMED_AT_K"+str(n)
    return True,"UNRESOLVED_ADD_NEXT_INDEPENDENT_SEED_BLOCK"

def freeze_contract():
    if OUT.exists():raise FileExistsError(OUT)
    freeze=SOURCE/"FROZEN_UPSTREAM_A_STATE_FORKS_TRAIN_ONLY.jsonl"
    previous=SOURCE/"POLICY_SAMPLED_4B_K8/SAMPLED_ACTUAL_FROZEN_4B_TRAJECTORIES.jsonl"
    assert freeze.exists() and previous.exists()
    OUT.mkdir(parents=True)
    record={
      "purpose":"fixed one-shot Action SFT then improved frozen-team CCE K8 vs adaptive K16/K32",
      "source_verified_native_witnesses":"artifacts/20261124_cce_downstream_sft_v1",
      "source_same_state_A_forks_SHA256":sha(freeze),
      "source_historical_K8_4B_rollouts_SHA256":sha(previous),
      "train_roots":["P201","P202","P203","P205","P206","P207","P208"],
      "distinct_A_states":14,
      "current_action_verdict":"both signed A package sends native VALID",
      "policy":"ONE shared Qwen3-4B LoRA fresh Action SFT r8/a16 rank, no role routing",
      "SFT_steps":160,"SFT_batch":2,"SFT_lr":2e-5,"SFT_seed":7,
      "SFT_EXCLUDED_upstream_A_SEND_action_labels":True,
      "no_param_tuning_based_on_new_CCE_results":True,
      "stage1_evaluate_downstream_capability_before_CCE":True,
      "stage2_seeds32":list(SEEDS32),"K8_old_seeds_reused_for_before_after":list(FIRST_SEEDS),
      "K8_policy":"screen only; local exact two-sided p<=0.05 reported but NEVER treated as confirmed training label",
      "adaptive_budget_looks":[8,16,32],
      "adaptive_acceptance_looks":[16,32],
      "two_sided_paired_exact_sign":True,
      "confidence_familywise_correction":"Bonferroni over 14 A states and 2 confirmatory looks",
      "confidence_familywise_alpha":ALPHA_FWER,
      "per_test_confirm_alpha":ALPHA_CONFIRM,
      "min_discordant_pairs":MIN_DISCORDANT,
      "min_both_safe_success_pair_for_cost":MIN_BOTH_SAFE_FOR_COST,
      "independent_seed_blocks_disjoint":True,
      "paired_CRN_caution":"same Random(seed) sequential draws, not per-role identical exogenous noise after paths diverge",
      "acceptance_requires_direction_in_both_seed_blocks":True,
      "K8_futility_stop":"both branches zero violation, zero safe success, zero deadline misses",
      "CI_first_hierarchy":["observed future CI risk","CI safe task success","cost only between jointly CI-safe completed trajectories"],
      "unresolved_observed_CI_risk_blocks_success_and_cost_labels":True,
      "UNKNOWN_blocks_hard_preference":True,
      "terminal_reward":"unchanged +1 safe complete, -1 actual CI violation, 0 safe incomplete",
      "Qstar_only_separate_offline_audit":True,
      "no_CCE_preferences_from_native_optimal_traces":True,
      "no_DEV_or_FINAL_label_used":True,
      "no_final_test_DG36_P211_P212_parsed":True,
      "stage3_train_only_if_confirmed_adaptive_preferences_exist":True,
      "cost_report":["rollouts","unique_role_observation_distributions","candidate_forward_calls","tokenized_model_prompt_tokens","actual_native_steps","GPU_wall"],
      "interpretation":"K8 exploratory vs adaptive confirmatory; not an anytime-valid confidence sequence",
    }
    (OUT/"PREREGISTERED_CCE_K8_ADAPTIVE.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_ADAPTIVE_PROTOCOL_FROZEN",json.dumps({"K8":8,"Kmax":32,"n_cells":14,
      "alpha_confirm":ALPHA_CONFIRM,"seed_sha":sha(OUT/"PREREGISTERED_CCE_K8_ADAPTIVE.json"),
      "no_new_A_preferences_from_Qstar":True},ensure_ascii=False))
if __name__=="__main__":freeze_contract()
