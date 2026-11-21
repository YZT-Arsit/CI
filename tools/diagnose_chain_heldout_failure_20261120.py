r"""Read-only, no environment/source edits: diagnose baseline CPU actor failure.

Recreate exact old training seed, inspect heldout A/B/source action logits
and publicly computable Horn closure. Verify what model genuinely sees.
"""
from pathlib import Path
from collections import Counter,defaultdict
from dataclasses import asdict
import json,sys,torch,numpy as np,random
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
import compare_kci_chain_methods_cpu as src
from chain_shared_actor_cpu import score_actions,actor_feature,stack,environment,digest,SharedHornActor
from kci.diverse_ci_branch_env import closure,FINAL
from solve_v31_longterm import native_best

OUT=ROOT/"artifacts"/"20261120_v31"/"CHAIN_CPU_HELDOUT_FAILURE_DIAG_01"
if OUT.exists():raise FileExistsError(OUT)
tr,cf,hi,cnt=src.build_expert_dataset(src.GROUP["train"])
cache=src.feature_cache(tr)
checks=[]
for mode in ("action_sft","cce_reference_dpo_proxy"):
    for seed in (7,):
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        net=SharedHornActor()
        opt=torch.optim.Adam(net.parameters(),lr=src.LR)
        for ep in range(src.WARMUP):src.optimize_sft(net,cache,opt)
        ref=__import__("copy").deepcopy(net).eval()
        cachedhi=src.feature_cache(hi);reff=src.ref_diff(ref,cachedhi)
        for ep in range(src.FINETUNE):
            if mode=="action_sft":src.optimize_sft(net,cache,opt)
            else:
                opt.zero_grad()
                loss=src.supervised_loss(net,cachedhi,"dpo",reff)+0.35*src.supervised_loss(net,cache,"sft")
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
        for split in ("train","dev","exploratory_test"):
            for row in src.GROUP[split]:
                env=environment(row)
                st=env.initial()
                st,ax=env.transition(st,"A_READ_VERIFIED_PACKAGES")
                assert ax["verdict"]=="VALID"
                aobs=env.observation(st)
                ascores=score_actions(net,aobs,["A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"]).detach().numpy()
                for package in ("PACKAGE_A","PACKAGE_B"):
                    st2,_=env.transition(st,"A_SEND_"+package)
                    for action in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                        st2,_=env.transition(st2,action)
                    obs=env.observation(st2)
                    x=stack([actor_feature(obs,a) for a in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")])
                    with torch.no_grad():
                        risk=net.infer(torch.maximum(x["known"],x["req"]),x["prem"],x["head"]).tolist()
                        logit=net(x).tolist()
                    base=set(z.split("::")[-1] for z in st2.native.delivered["R"] if z.startswith(env.native.case_subject+"::"))
                    graph=env.native.graph
                    exact=FINAL in closure(base|set(env.native.required_facts),graph)
                    q0=native_best(env.native,st2.native,"OPEN_PRIVATE_CHANNEL")["native_exact_Qstar"]
                    q1=native_best(env.native,st2.native,"SKIP_PRIVATE_CHANNEL")["native_exact_Qstar"]
                    rawfeat={
                        "known":x["known"][0].tolist(),"required":x["req"][0].tolist(),
                        "num_rules":int((x["head"][0].sum(-1)>0).sum()),
                        "soft_future_risk":float(risk[0]),
                        "B_open_logit":float(logit[0]),"B_skip_logit":float(logit[1]),
                        "B_pred":"OPEN" if logit[0]>logit[1] else "SKIP",
                        "true_risk_from_public_rule":exact,
                        "qgap":round(q0-q1,5),
                        "entry_role":obs["role"],"stage":obs["current_stage"]
                    }
                    checks.append({
                        "mode":mode,"seed":seed,"split":split,"world":row["world"],
                        "graph":row["graph"],"package":package,
                        "real_source_is_risky":package==row["related_package"],
                        "A_logits":{"PACKAGE_A":float(ascores[0]),"PACKAGE_B":float(ascores[1])},
                        **rawfeat
                    })
        print("DIAG",mode,"done",flush=True)
by=defaultdict(Counter)
for r in checks:
    group=(r["mode"],r["split"],r["graph"])
    by[group]["states"]+=1
    by[group]["B_correct"]+=int((r["B_pred"]=="OPEN")==r["true_risk_from_public_rule"])
    by[group]["soft_risk_always_gt05"]+=int(r["soft_future_risk"]>.5)
    by[group]["open_pct"]+=int(r["B_pred"]=="OPEN")
    by[group]["true_risk_count"]+=int(r["true_risk_from_public_rule"])
out={
    "status":"READ_ONLY_HELDOUT_CAUSAL_FEATURE_DIAG",
    "per_condition":checks,
    "by_split_graph":{str(k):dict(v) for k,v in by.items()},
}
OUT.mkdir()
(OUT/"DIAGNOSIS.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("RESULTS",json.dumps(out["by_split_graph"],ensure_ascii=False,indent=2),flush=True)
