r"""Paired low-data heldout graph ablation: optimal label BCE vs Q*-weight BCE.

Data are selected by TRAIN mechanism families only, both history variants
per selected mechanism. Identical subsets, init seeds, 160 updates, Adam LR.
Unseen graph results and actual Q* regret recorded. No test tuning.
"""
from pathlib import Path
from collections import defaultdict
from hashlib import sha256
import random,json,sys
import numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from train_v31_cpu_policy_probe import (
    get_rows,pack,NeuralRule,compute_native_terminal_returns,evaluate)
OUT=ROOT/"artifacts"/"20261120_v31"/"P0_LOW_DATA_LABEL_VS_QMARGIN_ABLATION"
if OUT.exists():raise FileExistsError(OUT)
torch.set_num_threads(2)
allrows=get_rows();compute_native_terminal_returns(allrows)
test=[x for x in allrows if x["split"]=="exploratory_test"]
dev=[x for x in allrows if x["split"]=="dev"]
train=[x for x in allrows if x["split"]=="train"]
by_mechanism=defaultdict(list)
for r in train:by_mechanism[r["case"]].append(r)
assert len(by_mechanism)==36 and all(len(v)==2 for v in by_mechanism.values())
group=defaultdict(list)
for id,v in by_mechanism.items():group[v[0]["graph"]].append(id)
assert len(group)==4 and all(len(ids)==9 for ids in group.values())
TEST=pack(test);DEV=pack(dev)
seeds=(7,1234,2025,42,999)
budgets=(4,8,16,36)
epochs=160
methods=("label_BCE","Qstar_weighted_BCE","shuffled_Qstar_BCE")
results=[]
for seed in seeds:
    rng=random.Random(seed)
    gkeys=sorted(group)
    for budget in budgets:
        # Stratified mechanism diversity, not cherry-picking performance.
        choices=[]
        order={key:rng.sample(group[key],len(group[key])) for key in gkeys}
        for depth in range(9):
            for key in gkeys:
                choices.append(order[key][depth])
        selected=choices[:budget]
        r=[row for name in selected for row in by_mechanism[name]]
        assert len(r)==2*budget
        x=pack(r)
        y=torch.tensor([v["y"] for v in r],dtype=torch.float32)
        margin=torch.tensor([v["margin"] for v in r],dtype=torch.float32)
        for method in methods:
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            net=NeuralRule()
            optim=torch.optim.Adam(net.parameters(),lr=0.03)
            if method=="label_BCE":weights=torch.ones_like(margin)
            elif method=="Qstar_weighted_BCE":weights=0.75+0.25*margin/margin.mean()
            else:
                perm=torch.as_tensor(np.random.default_rng(seed+budget+123).permutation(len(r)))
                m=margin[perm]
                weights=0.75+0.25*m/m.mean()
            snaps={}
            for step in range(epochs+1):
                if step in (20,40,80,160):
                    snaps[str(step)]={
                        "test":evaluate(net,TEST,test),
                        "dev":evaluate(net,DEV,dev)}
                if step==epochs:break
                optim.zero_grad()
                logits=net(x)
                loss=(F.binary_cross_entropy_with_logits(
                    logits,y,reduction="none")*weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=5)
                optim.step()
            results.append({
                "seed":seed,"training_mechanism_pairs":budget,
                "training_action_observations":2*budget,
                "training_graph_families":sorted(set(v["graph"] for v in r)),
                "method":method,"epochs":epochs,
                "same_train_subset_for_all_methods":True,
                "snapshots":snaps,
                "used_only_train_source_graphs":True})
        print("LOWDATA",seed,budget,
            [(x["method"],x["snapshots"]["160"]["test"]["accuracy"])
             for x in results[-3:]],flush=True)
aggregate={}
for b in budgets:
    aggregate[str(b)]={}
    for step in ("20","40","80","160"):
        aggregate[str(b)][step]={}
        for method in methods:
            rr=[x for x in results if x["method"]==method and
                x["training_mechanism_pairs"]==b]
            vals=[x["snapshots"][step]["test"]["accuracy"] for x in rr]
            aggregate[str(b)][step][method]={
                "mean_test_accuracy":round(float(np.mean(vals)),6),
                "sample_sd_test_accuracy":round(float(np.std(vals,ddof=1)),6),
                "seed_test_accuracies":vals,
                "mean_test_regret":round(float(np.mean([
                    x["snapshots"][step]["test"]["mean_native_regret_vs_Qstar"]
                    for x in rr])),6),
                "mean_test_flip_accuracy":round(float(np.mean([
                    x["snapshots"][step]["test"]["paired_flip_accuracy"]
                    for x in rr])),6),
            }
report={
    "status":"LOW_DATA_PAIRED_ACTION_LABEL_VS_QMARGIN_COMPLETED",
    "seeds":list(seeds),"budgets_mechanism_pairs":budgets,
    "methods":methods,"updates":epochs,
    "comparisons_are_same_seed_init_same_subset_same_optimizer":True,
    "training_and_eval_source_graph_split":True,
    "data_case_hash":sha256((ROOT/"artifacts"/"20261120_v31"/
      "V31_GATEA_FINAL_CANDIDATE"/"MODEL_OBSERVATIONS.jsonl").read_bytes()).hexdigest(),
    "not_Qpi_or_LLM":True,
    "aggregate":aggregate,
    "details":results,
}
OUT.mkdir(parents=True)
(OUT/"RESULTS.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("SUMMARY",json.dumps({
    "budget8_epoch160":aggregate["8"]["160"],
    "budget36_epoch160":aggregate["36"]["160"],
},ensure_ascii=False),flush=True)
