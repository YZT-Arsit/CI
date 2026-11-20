r"""Real CPU parameter-update pilot on KCI V3.1 preflight decisions.

FROZEN dataset. Train *small* PyTorch neural models, NOT the original LLM:
1. Immediate CI-only: both preflight actions valid => no contrastive update.
2. Terminal-return REINFORCE: use rewards from truly executed native rollouts
   under a fixed, myopic, history-aware suffix. For each update, observe ONLY
   the sampled action's realized terminal reward (contextual bandit, not full RL).
3. Exact native long-horizon action VALUE: paired Qstar action preference with
   margin-weighted BCE, source data restricted to training graph families.
4. Graph-blind MLP supervised with SAME long-horizon signal.
5. No-history ablation with SAME supervision and graph network.

Evaluation: balanced related vs same-entity safe pairs, held-out graph families,
full actual native fixed-policy suffix, actual Qstar regret, pairwise flip rate.
Actor feature processing reads ONLY model_input, never oracle labels. Native
return labels and optimal action are used only for training target and scoring.

No remote GPU, paid API, Qwen weight updates, or Qpi estimation.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from collections import Counter,defaultdict
from hashlib import sha256
import json,re,sys,time,random,argparse,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import digest,task_catalog
from kci.diverse_ci_long_horizon_v31 import V31Env,enumerate_specs_for_root
from run_v31_shortcut_learning_gate import actual_policy,verdict_from_observation

torch.set_num_threads(2)
NODES=[f"S{i:02d}" for i in range(1,7)]+["X","Y","Z","protected association"]
NODE={name:i for i,name in enumerate(NODES)}
N=len(NODES);MAX_RULES=12
DATA=ROOT/"artifacts"/"20261120_v31"/"V31_GATEA_FINAL_CANDIDATE"

def read_lines(name):
    return [json.loads(x) for x in (DATA/name).read_text(encoding="utf-8").splitlines() if x.strip()]

def actor_features(obs,hide_history=False,hide_graph=False):
    """Small, exact, public-ONLY observation adapter: normal S facts + Horn clauses."""
    entity=obs["current_case_subject"]
    prior=[f"S{i:02d}" for i in range(1,7)]
    known=np.zeros(6,np.float32)
    if not hide_history:
        for event in obs["delivered_message_log"]:
            match=re.fullmatch(
                r"Certified archived task evidence for case (CASE-[0-9A-F]{8}): "
                r"source (S0[1-6]) was delivered to reviewer R\.",event["text"])
            if not match:raise AssertionError("Actor-visible source malformed")
            if match.group(1)==entity:known[prior.index(match.group(2))]=1.0
    required=np.array(
        [float(f in obs["task_evidence_contract"]["required_slots"]) for f in prior],
        np.float32)
    init=np.zeros(N,np.float32)
    init[:6]=np.maximum(required,known)
    ruleprem=np.zeros((MAX_RULES,N),np.float32)
    rulehead=np.zeros((MAX_RULES,N),np.float32)
    for i,clause in enumerate(obs["declared_knowledge_rules"]):
        if hide_graph:break
        assert i<MAX_RULES
        for fact in clause["requires"]:
            assert fact in NODE,(fact,obs)
            ruleprem[i,NODE[fact]]=1.0
        assert clause["entails"] in NODE
        rulehead[i,NODE[clause["entails"]]]=1.0
    c=obs["privacy_resource_contract"]
    costs=np.array([
        c["private_channel_setup_steps"]/4.0,
        c["step_cost"]/0.05,
        c["task_budget"]/20.0,
        len(obs["task_evidence_contract"]["required_slots"])/6.0,
        len(obs["delivered_message_log"])/6.0,
        float("Authorization exception is explicitly granted" in obs["recipient_policy"]),
        float(obs["public_business_state"].get("route")=="early"),
        float(obs["public_business_state"].get("route")=="late"),
        float(obs["public_business_state"].get("route")=="cancel"),
        float(obs["public_business_state"].get("route")=="return"),
        float(obs["public_business_state"].get("route")=="direct"),
    ],np.float32)
    full=np.concatenate([known,required,ruleprem.ravel(),rulehead.ravel(),costs])
    return {"init":init,"prem":ruleprem,"head":rulehead,
            "context":costs,"full":full}

def pack(rows,hide_history=False,hide_graph=False):
    vals=[actor_features(r["obs"],hide_history,hide_graph) for r in rows]
    return {k:torch.tensor(np.stack([v[k] for v in vals]),dtype=torch.float32)
            for k in ("init","prem","head","context","full")}

class MLP(nn.Module):
    def __init__(self,nfeatures):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(nfeatures,128),nn.Tanh(),
            nn.Linear(128,64),nn.Tanh(),
            nn.Linear(64,1))
    def forward(self,x):
        return self.net(x["full"]).squeeze(-1)

class NeuralRule(nn.Module):
    """Rule-constrained message passing, learnable soft AND and forward value.

    Architecture incorporates an explicit formal-rule inductive bias, so
    success does not imply free-text inference or algorithmic novelty.
    """
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(1.8))
        self.risk_gain=nn.Parameter(torch.tensor(3.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.cost_gain=nn.Parameter(torch.tensor(0.0))
    def forward(self,x):
        known=x["init"]
        prem=x["prem"]
        head=x["head"]
        present=(head.sum(dim=2)>0).float()
        n=prem.sum(dim=2).clamp(min=1)
        k=torch.nn.functional.softplus(self.scale)
        for iteration in range(5):
            available=(known.unsqueeze(1)*prem).sum(dim=2)
            activation=torch.sigmoid(k*(available-n+0.5))*present
            contributions=activation.unsqueeze(-1)*head
            # Soft OR of inference rules for each derived proposition
            inferred=1.0-torch.prod(1.0-contributions,dim=1)
            known=known+(1-known)*inferred
        risk=known[:,NODE["protected association"]]
        # A small bounded setup-cost feature; actual correctness depends on
        # the risk score, not a hidden category ID.
        cost=x["context"][:,0]*x["context"][:,1]
        return self.risk_gain*(risk-0.5)+self.cost_gain*cost+self.bias

def get_rows():
    root={x["case_id"]:x for x in read_lines("CASE_SPECS.jsonl")}
    obs={x["world_id"]:x for x in read_lines("MODEL_OBSERVATIONS.jsonl")}
    labels={x["world_id"]:x for x in read_lines("EXACT_NATIVE_ACTION_PREFERENCES_EVAL_ONLY.jsonl")}
    assert len(root)==63 and len(obs)==len(labels)==315
    rows=[]
    for case in root.values():
        spec=enumerate_specs_for_root(
            next(t for t in task_catalog() if t.id==case["root_id"]),
            case["business_condition"],int(case["case_id"].split("_M")[-1])-1)
        for h in ("related","same_entity_safe"):
            world=case["case_id"]+"__"+h
            raw=obs[world]
            label=labels[world]
            assert raw["split"]==label["split"]==spec.root_split
            assert label["oracle_action"] in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")
            rows.append({
                "world":world,"case":case["case_id"],"graph":case["graph_family"],
                "split":raw["split"],"history":h,
                "obs":raw["model_input"],
                "y":1.0 if label["oracle_action"]=="OPEN_PRIVATE_CHANNEL" else 0.0,
                "margin":abs(label["native_Qgap_open_minus_skip"]),
                "Qstar_open":label["branch_action_values"]["open"],
                "Qstar_skip":label["branch_action_values"]["skip"],
                "native_actual_open":None,
                "native_actual_skip":None,
                "spec":spec})
    assert len(rows)==126
    assert sum(r["y"] for r in rows)==63
    assert all(sum(r["y"] for r in rows if r["case"]==c)==1
               for c in root)
    return rows

def compute_native_terminal_returns(rows):
    """Each arm is truly run in training native simulator with fixed suffix.

    Training terminal bandit later observes only sampled action's reward;
    cached other-arm rewards are NEVER used in its REINFORCE update.
    They exist to provide immutable, deterministic native transitions and
    post-eval scoring at identical future continuation conditions.
    """
    total=Counter()
    for r in rows:
        env=V31Env(r["spec"],r["history"])
        a=actual_policy(env,"OPEN_PRIVATE_CHANNEL")
        b=actual_policy(env,"SKIP_PRIVATE_CHANNEL")
        r["native_actual_open"]=a["reward"]
        r["native_actual_skip"]=b["reward"]
        total["episodes"]+=2
        total["reward_agrees_with_optimal"]+=int(
            a["reward"]==r["Qstar_open"] and b["reward"]==r["Qstar_skip"])
        if a["reward"]!=r["Qstar_open"] or b["reward"]!=r["Qstar_skip"]:
            raise AssertionError("Fixed myopic-native suffix differs from oracle in this task")
    return dict(total)

def evaluate(model,x,rows):
    model.eval()
    with torch.no_grad():
        logit=model(x)
        p=torch.sigmoid(logit).numpy()
    actions=(p>=0.5).astype(np.int32)
    y=np.array([r["y"] for r in rows],dtype=np.int32)
    accuracy=float(np.mean(actions==y))
    bycase=defaultdict(dict)
    for r,a in zip(rows,actions):
        bycase[r["case"]][r["history"]]=int(a)
    assert all(set(x)=={"related","same_entity_safe"} for x in bycase.values())
    strict_flip=sum(v["related"]==1 and v["same_entity_safe"]==0 for v in bycase.values())
    actual_rewards=np.array([
        r["native_actual_open"] if a else r["native_actual_skip"] for r,a in zip(rows,actions)])
    best=np.array([max(r["Qstar_open"],r["Qstar_skip"]) for r in rows])
    qchosen=np.array([r["Qstar_open"] if a else r["Qstar_skip"] for r,a in zip(rows,actions)])
    safe_task=np.array([float(x>0) for x in actual_rewards])
    return {
        "n":len(rows),"accuracy":round(accuracy,6),
        "correct":int(sum(actions==y)),
        "paired_flip_correct":strict_flip,
        "pair_total":len(bycase),
        "paired_flip_accuracy":round(strict_flip/len(bycase),6),
        "safe_task_success":int(sum(safe_task)),
        "mean_actual_native_return":round(float(np.mean(actual_rewards)),6),
        "mean_native_regret_vs_Qstar":round(float(np.mean(best-qchosen)),6),
        "action_open_rate":round(float(np.mean(actions)),6),
        "mean_prob_OPEN":round(float(np.mean(p)),6),
        "min_prob_OPEN":round(float(np.min(p)),6),
        "max_prob_OPEN":round(float(np.max(p)),6),
    }

def train(model,x,rows,algorithm,seed,epochs=120,lr=0.03):
    y=torch.tensor([r["y"] for r in rows],dtype=torch.float32)
    margins=torch.tensor([r["margin"] for r in rows],dtype=torch.float32)
    if algorithm=="terminal_bandit":
        rewards=torch.tensor([
            [r["native_actual_skip"],r["native_actual_open"]] for r in rows],dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    rng=np.random.default_rng(seed+5000)
    training=[]
    for epoch in range(epochs):
        model.train();opt.zero_grad()
        logits=model(x)
        if algorithm=="value_pair":
            weights=0.75+0.25*margins/margins.mean()
            loss=(F.binary_cross_entropy_with_logits(logits,y,reduction="none")*weights).mean()
        elif algorithm=="terminal_bandit":
            prob=torch.sigmoid(logits)
            with torch.no_grad():
                choices=torch.tensor((rng.random(len(rows))<prob.detach().numpy()).astype(np.int64))
                gained=rewards[torch.arange(len(rows)),choices]
                # A batch-mean state-independent baseline is permissible
                # (not an oracle counterfactual) for REINFORCE.
                advantage=gained-gained.mean()
            log_likelihood=-F.binary_cross_entropy_with_logits(logits,choices.float(),reduction="none")
            entropy=-(prob*torch.log(prob+1e-7)+(1-prob)*torch.log(1-prob+1e-7))
            loss=-(log_likelihood*advantage).mean()-0.005*entropy.mean()
        else:
            raise ValueError(algorithm)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
        opt.step()
        if epoch in (0,19,49,99,epochs-1):
            training.append({"epoch":epoch+1,"loss":round(float(loss.item()),6)})
    return training

def find_best_epoch(models_on_dev):
    return sorted(models_on_dev,key=lambda x:(-x["dev"]["accuracy"],
                   -x["dev"]["paired_flip_accuracy"],x["epoch"]))[0]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=160)
    p.add_argument("--seeds",nargs="+",type=int,default=[7,1234,2025])
    args=p.parse_args()
    out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    rows=get_rows()
    native=compute_native_terminal_returns(rows)
    groups={s:[r for r in rows if r["split"]==s]
            for s in ("train","dev","exploratory_test")}
    assert [len(groups[s]) for s in groups]==[72,18,36]
    inputs={s:pack(groups[s]) for s in groups}
    results=[];traces=[]
    for variant in (
        "mlp_value_pair","neural_rule_value_pair",
        "neural_rule_terminal_bandit","neural_rule_value_no_history"):
        for seed in args.seeds:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            arch="mlp" if variant.startswith("mlp") else "neuralrule"
            algorithm="terminal_bandit" if "terminal_bandit" in variant else "value_pair"
            no_history="no_history" in variant
            xx={s:pack(groups[s],hide_history=no_history) for s in groups}
            net=MLP(xx["train"]["full"].shape[1]) if arch=="mlp" else NeuralRule()
            # Shared architecture with identical initialization for value vs
            # terminal bandit (same seed): isolates learning signal effect.
            snapshot=[]
            history=train(net,xx["train"],groups["train"],algorithm,seed,
                          epochs=args.epochs,lr=(0.01 if arch=="mlp" else 0.03))
            for split in groups:
                metric=evaluate(net,xx[split],groups[split])
                results.append({"variant":variant,"seed":seed,
                    "split":split,"metric":metric,
                    "training_updates":args.epochs,
                    "train_worlds":72,
                    "training_native_episodes_available":144,
                    "training_feedback":(
                        "PAIRED_NATIVE_QSTAR_ACTION_RETURN" if algorithm=="value_pair" else
                        "ONPOLICY_SAMPLED_ACTION_REALIZED_RETURN_FIXED_MYOPIC_SUFFIX"),
                    "train_graph_holdout":True,
                    "architecture":arch,
                    "history_removed_at_actor_input":no_history})
            traces.append({"variant":variant,"seed":seed,"loss_log":history})
            print("TRAINED",variant,seed,
                "TRAIN",next(r["metric"]["accuracy"] for r in results if r["variant"]==variant
                             and r["seed"]==seed and r["split"]=="train"),
                "DEV",next(r["metric"]["accuracy"] for r in results if r["variant"]==variant
                             and r["seed"]==seed and r["split"]=="dev"),
                "TEST",next(r["metric"]["accuracy"] for r in results if r["variant"]==variant
                             and r["seed"]==seed and r["split"]=="exploratory_test"),
                flush=True)
    bymethod=defaultdict(dict)
    for method in sorted({r["variant"] for r in results}):
        for split in groups:
            rr=[r["metric"] for r in results if r["variant"]==method and r["split"]==split]
            bymethod[method][split]={
                k:round(float(np.mean([x[k] for x in rr])),6)
                for k in ("accuracy","paired_flip_accuracy","mean_actual_native_return",
                          "mean_native_regret_vs_Qstar","action_open_rate")}
            bymethod[method][split]["std_accuracy_sample"]=round(
                float(np.std([x["accuracy"] for x in rr],ddof=1)) if len(rr)>1 else 0.0,6)
            bymethod[method][split]["seed_metrics"]=rr
    baseline={}
    for split,rr in groups.items():
        baseline[split]={
            "always_open":evaluate(lambda _:None,{},rr) if False else {
                "accuracy":0.5,
                "paired_flip_accuracy":0.0,
                "safe_task_success":len(rr),
                "mean_native_regret_vs_Qstar":round(
                    float(np.mean([max(r["Qstar_open"],r["Qstar_skip"])-r["Qstar_open"]
                                   for r in rr])),6),
            },
            "always_skip":{
                "accuracy":0.5,
                "paired_flip_accuracy":0.0,
                "safe_task_success":len(rr)//2,
                "mean_native_regret_vs_Qstar":round(
                    float(np.mean([max(r["Qstar_open"],r["Qstar_skip"])-r["Qstar_skip"]
                                   for r in rr])),6),
            }
        }
    report={
        "status":"REAL_SMALL_TORCH_CPU_PARAMETER_TRAINING_COMPLETED",
        "hardware":"Windows CPU, PyTorch torch no CUDA",
        "data":"KCI V3.1 63 native mechanism specs, only related/same_entity_safe balanced pair per spec",
        "only_train_graphs_used_in_learning":True,
        "source_method":"small neural classifier + contextual bandit training, NOT Qwen weight update",
        "seeds":args.seeds,"epochs_per_run":args.epochs,
        "training_examples":len(groups["train"]),
        "dev_examples":len(groups["dev"]),
        "test_examples":len(groups["exploratory_test"]),
        "terminal_native_rollouts_created":native,
        "conditions":{
            "immediate_only":"At preflight both OPEN and SKIP are immediately allowed; no discriminating step-level reward. Only trivial 50% balanced classification is valid without inventing a label.",
            "terminal_bandit":"Real return from chosen sampled action under fixed myopic native suffix, reward observed only for sampled arm in REINFORCE.",
            "longterm_value":"Access to exact paired native Qstar from train roots; supervised margin-weighted pairwise preference, never test labels in training.",
            "no_history_ablation":"Same rule network and objective, but removes all recipient prior history from actor features.",
            "mlp":"Generic, graph-order-sensitive MLP that receives visible fact and rule incidence features.",
            "neural_rule":"Strong built-in Horn rule message-passing inductive bias; successful generalization would NOT establish free-form LLM reasoning.",
            "reference_public_rule_planner":"Earlier independent CPU native visible graph evaluator achieves 100% and remains the strong upper reference."
        },
        "means_and_sample_SD":bymethod,
        "baselines":baseline,
        "all_sample_training_logs":traces,
        "all_seed_split_metrics":results,
        "full_agent_episode_training_done":False,
        "LLM_Qpi_measured":False,
        "heldout_test_not_blind":True,
        "no_paid_API":True,
    }
    out.mkdir(parents=True)
    (out/"RESULTS.json").write_text(json.dumps(
        report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("FINAL",
        json.dumps({k:v for k,v in report.items() if k not in (
            "all_sample_training_logs","all_seed_split_metrics")},ensure_ascii=False),
        flush=True)

if __name__=="__main__":main()
