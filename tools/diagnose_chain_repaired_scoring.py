"""Diagnose if correctly inferred public Horn closure actually affects B logits.

Train original SFT/CCE on frozen original train split, with ONLY fixed
zero-preserving Horn infer. Audit A-package / B action logits and label
balance at every graph root. Read-only analysis; no test tuning labels.
"""
from pathlib import Path
from collections import Counter,defaultdict
from copy import deepcopy
import sys,random,json,torch,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
import compare_kci_chain_methods_cpu_fixed as X
from chain_shared_actor_cpu_fixed import SharedHornActor,actor_feature,score_actions,stack,environment
from kci.diverse_ci_branch_env import FINAL,closure
from solve_v31_longterm import native_best
OUT=ROOT/"artifacts"/"20261120_v31"/"CHAIN_CPU_FIXED_HORN_HEAD_DIAGNOSTIC_02"
if OUT.exists():raise FileExistsError(OUT)
routine,more,high,counts=X.build_expert_dataset(X.GROUP["train"])
info={
    "train_sft_action_choice_by_role":dict(Counter((p["actor"],p["chosen"]) for p in routine)),
    "train_extra_sft_action_choice_by_role":dict(Counter((p["actor"],p["chosen"]) for p in more)),
    "train_cce_pref_by_actor":dict(Counter((p["actor"],p["chosen"],p["rejected"]) for p in high)),
}
source=[]
for method in ("action_sft","cce_reference_dpo_proxy"):
    seed=7
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    net=SharedHornActor()
    std=X.feature_cache(routine);highcache=X.feature_cache(high)
    opt=torch.optim.Adam(net.parameters(),lr=X.LR)
    for _ in range(X.WARMUP):X.optimize_sft(net,std,opt)
    baseline=X.ref_diff(net,highcache)
    for _ in range(X.FINETUNE):
        if method=="action_sft":X.optimize_sft(net,std,opt)
        else:
            opt.zero_grad()
            loss=X.supervised_loss(net,highcache,"dpo",baseline)+0.35*X.supervised_loss(net,std)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
    for split in ("train","dev","exploratory_test"):
        for row in X.GROUP[split]:
            env=environment(row)
            s=env.initial();s,_=env.transition(s,"A_READ_VERIFIED_PACKAGES")
            aview=env.observation(s)
            with torch.no_grad():alogits=score_actions(net,aview,["A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"]).tolist()
            for pkg in ("PACKAGE_A","PACKAGE_B"):
                st,_=env.transition(s,"A_SEND_"+pkg)
                for act in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                    st,_=env.transition(st,act)
                obs=env.observation(st)
                xx=stack([actor_feature(obs,"OPEN_PRIVATE_CHANNEL"),actor_feature(obs,"SKIP_PRIVATE_CHANNEL")])
                with torch.no_grad():
                    risk=net.infer(torch.maximum(xx["known"],xx["req"]),xx["prem"],xx["head"])
                    logits=net(xx)
                true=FINAL in closure(set(z.split("::")[-1] for z in st.native.delivered["R"])|set(env.native.required_facts),env.native.graph)
                assert abs(float(risk[0].item())-float(true))<1e-6
                source.append({
                    "method":method,"split":split,"graph":row["graph"],"world":row["world"],
                    "pkg":pkg,"risk":true,"soft_risk":float(risk[0]),
                    "open_score":float(logits[0]),"skip_score":float(logits[1]),
                    "pred_open":bool(logits[0]>logits[1]),
                    "raw_source_feature":xx["src"][0].tolist(),
                    "known":xx["known"][0].tolist(),
                    "required":xx["req"][0].tolist(),
                    "a_logits":alogits,
                })
    print("TRAINED",method,flush=True)
aggregated={}
for method in ("action_sft","cce_reference_dpo_proxy"):
    for split in ("train","dev","exploratory_test"):
        rows=[x for x in source if x["method"]==method and x["split"]==split]
        risks=[x for x in rows if x["risk"]]
        benign=[x for x in rows if not x["risk"]]
        aggregated[method+"_"+split]={
            "risky_n":len(risks),"benign_n":len(benign),
            "open_at_risky":sum(x["pred_open"] for x in risks),
            "skip_at_safe":sum(not x["pred_open"] for x in benign),
            "risky_mean_score_gap":round(sum(x["open_score"]-x["skip_score"] for x in risks)/len(risks),4),
            "safe_mean_score_gap":round(sum(x["open_score"]-x["skip_score"] for x in benign)/len(benign),4),
        }
OUT.mkdir()
(OUT/"SCORING.json").write_text(json.dumps({"data":source,"aggregation":aggregated,
    "training_label_distribution":{k:{str(kk):vv for kk,vv in v.items()} for k,v in info.items()}},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("AGG",json.dumps(aggregated,ensure_ascii=False,indent=2),flush=True)
print("TRAIN_SUPERVISION",str(info),flush=True)
