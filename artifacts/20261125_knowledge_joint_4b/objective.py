"""Exact finite-menu objectives. Each dictionary of logits belongs to one current policy."""
import gzip, json
from fractions import Fraction
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
RHO=1.0
LAMBDA=0.25
FIELDS=("stsr","risk","direct","inferred","safe_abort","unknown")

def read(name):
    with gzip.open(HERE/(name+".json.gz"),"rt",encoding="utf8") as f:return json.load(f)

def dump(path,obj):
    path=Path(path);tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf8");tmp.replace(path)

def tags(w):
    return list(dict.fromkeys([w["head"]]+[b["tag"] for bs in w["paths"].values() for b in bs]))

def leaf(v):
    if v is None or any(v.get(k) is None for k in ("stsr","direct","inferred","abort")):
        return [0,0,0,0,0,1]  # known event mass and explicit residual UNKNOWN
    risk=bool(v["direct"] or v["inferred"])
    return [v["stsr"],risk,v["direct"],v["inferred"],v["abort"] and not risk,0]

def policy(z,greedy):
    if greedy:
        p=torch.zeros_like(z);p[z.argmax()]=1
        return p
    return z.softmax(0)

def accepted_log(z,actions,valid):
    ids=[actions.index(a) for a in valid]
    return z[ids].logsumexp(0)-z.logsumexp(0)

def compute(g,obs,scores,arm=None,greedy=False,norm="original"):
    w=g["worlds"][norm];h=w["head"];actions=obs[h]["actions"];z=scores[h]
    pa=policy(z,greedy); values=[]; ceilings=[];goodmass={};goodlog={}
    targets=[-accepted_log(z,actions,w["accept"])] if w["accept"] else []
    opportunity=[]; available=[]
    for a in actions:
        rows=w["paths"].get(a,[]);mass=Fraction(0)
        val=z.new_zeros(len(FIELDS));cap=0.;leg=z.new_zeros(());support=0.
        for b in rows:
            q=Fraction(b["q"]);mass+=q
            t=b["tag"];menu=obs[t]["actions"];pb=policy(scores[t],greedy)
            table=z.new_tensor([leaf(b["leaves"].get(k)) for k in menu])
            v=pb@table
            val=val+float(q)*v
            if b["open"]:
                cap+=float(q);support+=float(q);leg=leg+float(q)*v[0]
            if b["accept"] and not table[:,-1].any().item():
                al=accepted_log(scores[t],menu,b["accept"])
                goodlog[t]=al;goodmass[t]=pb[[menu.index(k) for k in b["accept"]]].sum()
                targets.append(-al)
        assert 0<=mass<=1
        val=val+z.new_tensor([0,0,0,0,0,float(1-mass)])
        values.append(val);ceilings.append(cap);opportunity.append(leg);available.append(support)
    totals=pa@torch.stack(values)
    metrics=dict(zip(FIELDS,totals.unbind()))
    if arm and metrics["unknown"].item()>0:raise ValueError("Incomplete native root: UNKNOWN")
    metrics["regret"]=max(ceilings)-pa@z.new_tensor(ceilings)
    # Force all upstream histories uniformly, conditional on a verified safe-completion opportunity.
    metrics["legal"]=sum(opportunity)/sum(available) if sum(available) else None
    joint=hist=None
    if norm=="original" and g["matches"]:
        joint=torch.stack([goodmass[l]*goodmass[r] for l,r in g["matches"]]).mean()
        hist=-torch.stack([goodlog[l]+goodlog[r] for l,r in g["matches"]]).mean()
    metrics["history"]=joint
    if norm=="original":
        hold=g.get("hold")
        metrics["irrelevant"]=goodmass[hold[0]]*goodmass[hold[1]] if hold else None
    if norm=="authorized":
        bad=w["paths"][g["pair"][1]][0]
        metrics["restoration"]=goodmass[bad["tag"]] if bad["open"] else None
    loss=None
    if arm=="A":loss=torch.stack(targets).mean()
    if arm in ("B","C"):
        loss=-metrics["stsr"]+RHO*metrics["risk"]
        if arm=="C" and hist is not None:loss=loss+LAMBDA*hist
    return loss,metrics

def scalars(m):
    return {k:None if x is None else float(x.detach()) for k,x in m.items()}
