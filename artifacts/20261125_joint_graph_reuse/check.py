"""Exact two-actor graph checks on existing native branches and saved 4B scores."""
import gzip
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ART = OUT.parent


def read(p):
    with (gzip.open if p.suffix == ".gz" else open)(p, "rt", encoding="utf8") as f:
        return json.load(f)


def reuse():
    folder=ART/"20261125_knowledge_joint_4b"
    graphs=read(folder/"groups.json.gz")
    obs=read(folder/"observations.json.gz")
    gen=torch.Generator().manual_seed(7)
    z={k:torch.randn(len(v["actions"]),generator=gen,dtype=torch.float64,requires_grad=True) for k,v in obs.items()}
    out={}
    for domain in sorted({g["domain"] for g in graphs}):
        worlds=[w for g in graphs if g["domain"]==domain for w in g["worlds"].values()]
        def run(shared):
            prob={}; table={}; value={}; totals=[];counts=Counter()
            def pi(t):
                if not shared or t not in prob:
                    prob[t]=z[t].softmax(0);counts["policy_evals"]+=1
                return prob[t]
            for w in worlds:
                tag=w["head"];p=pi(tag);terms=[]
                assert set(w["paths"])==set(obs[tag]["actions"])
                for a in obs[tag]["actions"]:
                    bs=w["paths"][a];assert sum(Fraction(b["q"]) for b in bs)==1
                    for b in bs:
                        t=b["tag"];menu=obs[t]["actions"];ls=b["leaves"]
                        assert set(menu)==set(ls)
                        rows=[]
                        for k in menu:
                            v=ls[k];assert all(isinstance(v[n],bool) for n in ("stsr","direct","inferred","abort"))
                            rows.append((v["stsr"],v["direct"] or v["inferred"]))
                        key=tuple(rows)
                        if not shared or key not in table:
                            table[key]=torch.tensor(key,dtype=torch.float64);counts["numeric_tables"]+=1
                        if not shared or (t,key) not in value:
                            value[t,key]=pi(t)@table[key];counts["downstream_values"]+=1
                        terms.append(p[obs[tag]["actions"].index(a)]*float(Fraction(b["q"]))*value[t,key])
                totals.append(torch.stack(terms).sum(0))
            vals=torch.stack(totals);metrics=vals.mean(0)
            grads=[torch.autograd.grad(x,list(z.values()),allow_unused=True,retain_graph=True) for x in metrics]
            return vals.detach(),grads,counts
        a,da,ca=run(False);b,db,cb=run(True)
        err=(a-b).abs().max().item()
        grad=max((x-y).abs().max().item() for u,v in zip(da,db) for x,y in zip(u,v) if x is not None and y is not None)
        assert all((x is None)==(y is None) for u,v in zip(da,db) for x,y in zip(u,v))
        assert max(err,grad)<1e-10
        out[domain]=dict(worlds=len(worlds),plain=ca,shared=cb,max_value_error=err,max_gradient_error=grad)
    return out


def mixture():
    old=ART/"20261125_continuation_4b";s1=ART/"20261125_process_policy/s1"
    gs=read(old/"graphs.gz");obs=read(old/"observations.gz")
    def probs(name):
        scores=read(s1/name)["scores"]
        def soft(x):
            x=np.exp(x-x.max(axis=1,keepdims=True));return x/x.sum(axis=1,keepdims=True)
        return {k:soft(np.asarray(v)) for k,v in scores.items()}
    p=probs("step_000.json");end=probs("step_1024.json")
    d={k:end[k]-v for k,v in p.items()}; coeffs={}
    contrasts={}
    for g in gs:
        w=g["worlds"]["original"];h=w["head"];c=np.zeros((3,2))
        # Paired cache diagnostic, separate from the pre-specified policy selection.
        branches=[w["paths"][a] for a in g["pair"]]
        menus=[sorted(obs[b["tag"]]["actions"]) for b in branches]
        assert menus[0]==menus[1]
        labels=[np.array([b["classes"][m] for m in menus[0]]) for b in branches]
        pay0,pay1=[np.column_stack((v==2,v%2==1)).astype(float) for v in labels]
        p0,p1=[end[b["tag"]] for b in branches]
        delta=p1@(pay1-pay0)+(p1-p0)@pay0
        assert np.allclose(p1@pay1-p0@pay0,delta,atol=1e-12,rtol=0)
        stats=contrasts.setdefault(g["partition"],Counter())
        stats["roots"]+=1
        stats["equal_task_vectors"]+=int(np.array_equal(labels[0]//2,labels[1]//2))
        stats["different_risk_vectors"]+=int(np.any((labels[0]%2)!=(labels[1]%2)))
        stats["action_pairs"]+=len(menus[0])
        stats["changed_SR_rows"]+=int(np.any(pay0!=pay1,axis=1).sum())
        stats["policy_response_affects_value"]+=int(np.max(abs((p1-p0)@pay0))>1e-10)
        for j,a in enumerate(sorted(obs[h]["actions"])):
            b=w["paths"][a];t=b["tag"];menu=sorted(obs[t]["actions"])
            assert set(menu)==set(b["classes"])
            labels=np.array([b["classes"][m] for m in menu]);pay=np.stack((labels==2,labels%2==1),axis=1)
            base=p[t]@pay;delta=d[t]@pay
            def term(x,v):return (x[:,j,None]*v).mean(axis=0)
            c+=np.stack((term(p[h],base),term(d[h],base)+term(p[h],delta),term(d[h],delta)))
        coeffs[g["partition"],g["root"]]=c
    co={k:np.mean([v for (s,_),v in coeffs.items() if s==k],axis=0) for k in ("train","dev")}
    assert len(coeffs)==160 and len({r for _,r in coeffs})==160
    def at(c,t):return np.array([1,t,t*t])@c
    c=co["train"];result={}
    for name,cap in [("zero",0.),("relative",c[0,1])]:
        candidates=[0.,1.]
        for r in np.roots([c[2,1],c[1,1],c[0,1]-cap]):
            if abs(r.imag)<1e-12 and 0<=r.real<=1:candidates.append(float(r.real))
        if c[2,0] and 0< -c[1,0]/(2*c[2,0])<1:candidates.append(-c[1,0]/(2*c[2,0]))
        valid=[t for t in candidates if at(c,t)[1]<=cap+1e-12]
        result[name]=dict(cap=cap,t=max(valid,key=lambda t:(at(c,t)[0],-t)) if valid else None)
    result["coefficients"]={s:v.tolist() for s,v in co.items()}
    for label,t in [("initial",0.),("S1",1.),("selected",result["relative"]["t"])]:
        result[label]={s:at(v,t).tolist() for s,v in co.items()}
    result["posthoc_history_comparison"]=contrasts
    return result


if __name__=="__main__":
    result=dict(reuse=reuse(),policy=mixture())
    text=json.dumps(result,indent=2,allow_nan=False)
    (OUT/"RESULTS.json").write_text(text,encoding="utf8")
    print(text)
