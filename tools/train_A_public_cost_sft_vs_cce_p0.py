r"""P0 focused A-action learning: cost observability, oracle label vs Q magnitude.

Only certified RecordsAgent model-input (A) is encoded, two legal A actions
per *same* source-tool state. Public Horn rules, future required checks,
private setup steps, source costs are available. No native future search or
hidden preferred actions in inference. Actual native Q* references are used
ONLY to produce train labels/weight and evaluate held-out regret.

14 non-tie source worlds in train, 6 non-tie in exploratory tests (+2 ties).
Controls:
 - exact public visible planner WITHOUT learning;
 - risk-only cost features (risk + setup, no additional attestation counts);
 - full visible cost decomposition including #attestations (strong planning prior).
 - action label SFT, same-label Qgap-weighted SFT, margin-shuffled SFT, and
   reference-relative Qgap-weighted pair DPO proxy.
5 paired seeds, identical initial same model and equal 80 updates per method.
No official CCE method or LLM training; exploratory split has been inspected.
"""
from pathlib import Path
from hashlib import sha256
import json,sys,random,time
from collections import defaultdict,Counter
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import environment
from kci.upstream_downstream_chain_pilot_public_contract import ChainPilotEnvPublicContract

OUT=ROOT/"artifacts"/"20261120_v31"/"A_PUBLIC_COST_SFT_CCE_PAIRED_P0_01"
if OUT.exists():raise FileExistsError(OUT)
torch.set_num_threads(2)
TYPES=("PACKAGE_A","PACKAGE_B")
def closure_visible(facts,clauses):
    x=set(facts)
    for _ in range(len(clauses)+2):
        delta={r["entails"] for r in clauses if set(r["requires"])<=x}
        if delta<=x:break
        x|=delta
    return x
def feature(obs,package,full=True):
    assert package in obs["source_bundle_tool_output"]
    p=obs["source_bundle_tool_output"][package]
    c=obs["privacy_resource_contract"]
    known=set(p["fact_slots"])
    req=obs["downstream_required_business_checks"]
    rules=obs["public_inference_rules"]
    allow="Authorization exception is explicitly granted" in obs["privacy_policy"]
    future_risk=("protected association" in closure_visible(known|set(req),rules)) and not allow
    ncert=0
    for f in req:
        if not allow and "protected association" in closure_visible(known|{f},rules):
            ncert+=1 # private attestation, fact not sent to R
        else:known.add(f)
    # The network is NOT given the true Qstar labels or desired action.
    # The risk and anticipated verification count are legitimately computed
    # from the model-observable rules, checks and sources; these features
    # impose a strong exact-planning inductive bias.
    return np.array([
       float(p["cost_steps"])/6,
       float(future_risk)*(float(c["private_channel_setup_steps"])-1)/6,
       float(ncert)/4 if full else 0.,
       float(future_risk),
       float(len(req))/4,
    ],np.float32)

records={s:[] for s in GROUP}
for split,worlds in GROUP.items():
    for row in worlds:
        E=environment(row)
        e=ChainPilotEnvPublicContract(E.spec,E.swap,E.source_mode,E.safe_surcharge)
        st=e.initial();st,ver=e.transition(st,"A_READ_VERIFIED_PACKAGES")
        assert ver["verdict"]=="VALID"
        o=e.observation(st)
        assert "privacy_resource_contract" in o
        q=row["upstream_A_values_given_exact_B_continuation"]
        gap=q["PACKAGE_A"]-q["PACKAGE_B"]
        r={"world":row["world"],"split":split,"graph":row["graph"],
           "qA":q["PACKAGE_A"],"qB":q["PACKAGE_B"],"gap":gap,
           "strict":abs(gap)>1e-8,
           "features_full":np.stack([feature(o,x,True) for x in TYPES]),
           "features_partial":np.stack([feature(o,x,False) for x in TYPES]),
           "observation_sha256":__import__("kci.diverse_ci_branch_env",fromlist=["digest"]).digest(o)}
        assert np.isfinite(r["features_full"]).all()
        records[split].append(r)
assert [len(records[s]) for s in ("train","dev","exploratory_test")]==[16,4,8]
assert [sum(v["strict"] for v in records[s]) for s in ("train","dev","exploratory_test")]==[14,4,6]

def metrics(net,split,full):
    rows=records[split]
    outputs=[]
    strict_total=correct_strict=tie=all_correct=0
    regrets=[]
    for r in rows:
        x=torch.tensor(r["features_full" if full else "features_partial"])
        with torch.no_grad():
            logits=net(x).squeeze(-1)
            prob=torch.softmax(logits,dim=0)[0].item()
        selected="PACKAGE_A" if prob>=0.5 else "PACKAGE_B"
        best=max(r["qA"],r["qB"])
        val=r["qA"] if selected=="PACKAGE_A" else r["qB"]
        regret=round(best-val,8)
        ok=regret<1e-7
        all_correct+=int(ok)
        regrets.append(regret)
        if r["strict"]:
            strict_total+=1
            correct_strict+=int(ok)
        else:tie+=1
        outputs.append({"world":r["world"],"p_PACKAGE_A":round(prob,7),
            "prediction":selected,"correct_or_tie":ok,
            "true_Qstar_gap_EVAL_ONLY":r["gap"],"regret":regret})
    return {
        "all_correct_including_ties":all_correct,"all_total":len(rows),
        "strict_correct":correct_strict,"strict_total":strict_total,
        "ties":tie,"mean_regret":round(float(np.mean(regrets)),7),
        "chosen_probabilities_and_eval_Qgaps":outputs,
    }

def learn(mode,seed,full):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=nn.Linear(5,1,bias=False)
    with torch.no_grad():model.weight.normal_(mean=0,std=.025)
    ref=model.weight.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    tr=[r for r in records["train"] if r["strict"]]
    inp=torch.tensor(np.stack([r["features_full" if full else "features_partial"] for r in tr]))
    q=torch.tensor([r["gap"] for r in tr],dtype=torch.float32)
    gold=torch.tensor([0 if r["gap"]>0 else 1 for r in tr],dtype=torch.long)
    margins=q.abs()
    weights=margins/margins.mean()
    if mode=="Qgap_shuffled_SFT":
        weights=weights[torch.tensor(np.random.default_rng(seed+731).permutation(len(tr)))]
    snapshots=[]
    for epoch in range(81):
        if epoch in (0,1,2,5,10,20,40,80):
            snapshots.append({
                "epoch":epoch,"train":metrics(model,"train",full),
                "dev":metrics(model,"dev",full),
                "exploratory_test":metrics(model,"exploratory_test",full)})
        if epoch==80:break
        optimizer.zero_grad()
        logits=model(inp).squeeze(-1)
        if mode=="optimal_action_SFT":
            loss=F.cross_entropy(logits,gold)
        elif mode in ("Qgap_weighted_SFT","Qgap_shuffled_SFT"):
            loss=(F.cross_entropy(logits,gold,reduction="none")*weights).mean()
        elif mode=="Qgap_reference_pair_DPO_proxy":
            delta=logits[torch.arange(len(tr)),gold]-logits[torch.arange(len(tr)),1-gold]
            with torch.no_grad():
                rlogits=inp.matmul(ref.t()).squeeze(-1)
                delta_ref=rlogits[torch.arange(len(tr)),gold]-rlogits[torch.arange(len(tr)),1-gold]
            loss=-(weights*F.logsigmoid(2*(delta-delta_ref))).mean()
        else:raise ValueError(mode)
        loss.backward();optimizer.step()
    return {"mode":mode,"seed":seed,"full_visible_cert_count":full,
       "checkpoints":snapshots,
       "learned_weights":model.weight.detach().view(-1).tolist(),
       "training_examples_strict_only":len(tr),
       "training_updates":80,
       "all_trained_from_same_actor_observation_and_seed":True,
       "train_truth_only_target":True}

results=[]
for full in (False,True):
    for mode in ("optimal_action_SFT","Qgap_weighted_SFT","Qgap_shuffled_SFT",
                 "Qgap_reference_pair_DPO_proxy"):
        for seed in (7,1234,2025,42,999):
            record=learn(mode,seed,full)
            results.append(record)
            last=record["checkpoints"][-1]
            print("P0_A",full,mode,seed,
                  "TEST",last["exploratory_test"]["strict_correct"],
                  "/",last["exploratory_test"]["strict_total"],
                  "RET",last["exploratory_test"]["mean_regret"],
                  "TRAIN",last["train"]["strict_correct"],flush=True)
aggregate=defaultdict(dict)
for full in (False,True):
    for mode in ("optimal_action_SFT","Qgap_weighted_SFT","Qgap_shuffled_SFT",
                 "Qgap_reference_pair_DPO_proxy"):
        group=[r for r in results if r["mode"]==mode and r["full_visible_cert_count"]==full]
        aggregate[str(full)+"__"+mode]={
            "test_strict_correct_per_seed":[r["checkpoints"][-1]["exploratory_test"]["strict_correct"] for r in group],
            "test_all_correct_per_seed":[r["checkpoints"][-1]["exploratory_test"]["all_correct_including_ties"] for r in group],
            "test_mean_regret_per_seed":[r["checkpoints"][-1]["exploratory_test"]["mean_regret"] for r in group],
            "test_initial_strict_correct_per_seed":[r["checkpoints"][0]["exploratory_test"]["strict_correct"] for r in group],
            "train_strict_correct_per_seed":[r["checkpoints"][-1]["train"]["strict_correct"] for r in group],
            "epoch_to_zero_regret_in_test_first_checkpoint":[next((m["epoch"] for m in r["checkpoints"] if m["exploratory_test"]["mean_regret"]==0),None) for r in group]
        }
record={
 "status":"UPSTREAM_A_LABEL_VS_QSTAR_MAGNITUDE_SMALL_CPU_PAIRED_ABLATION",
 "source_DAG_holdout":"same original split, 4 train graph families and 2 developer-inspected exploratory graphs",
 "train_strict_action_states":14,"dev_strict_states":4,
 "test_strict_states":6,"test_ties":2,
 "same_actor_input_and_optimizer_per_feature_variant":True,
 "features_from_trusted_actor_visible_rules_and_costs_only":True,
 "full_feature_is_strong_public_planning_prior_not_language_learning":True,
 "original_frozen_V31_env_unchanged":True,
 "original_R_history_generated_by_actual_upstream_tool_delivery":True,
 "not_GPT_Qpi_or_full_shared_model_optimization":True,
 "seed_matched":True,"num_seeds":5,
 "aggregate":dict(aggregate),"all_seed_checkpoints":results,
}
OUT.mkdir(parents=True)
(OUT/"P0_A_SFT_VS_QSTAR.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("SUMMARY",json.dumps(record["aggregate"],ensure_ascii=False,indent=2),flush=True)
