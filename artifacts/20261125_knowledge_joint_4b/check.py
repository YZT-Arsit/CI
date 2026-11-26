"""Independent native leaf enumeration and shared-weight directional derivative checks."""
import copy, math
from fractions import Fraction
import torch
from objective import HERE,read,dump,tags,compute,FIELDS,leaf

def main():
    groups,obs=read("groups"),read("observations")
    torch.manual_seed(7)
    feature={t:torch.randn(len(r["actions"]),5,dtype=torch.double) for t,r in obs.items()}
    theta=torch.randn(5,dtype=torch.double,requires_grad=True)
    errors=[];rows=0;gradients=[]
    for g in groups:
        for norm,w in g["worlds"].items():
            z={t:feature[t]@theta for t in tags(w)}
            _,m=compute(g,obs,z,norm=norm)
            probs={t:z[t].detach().softmax(0).tolist() for t in z}
            ref=[0.]*len(FIELDS)
            for ia,a in enumerate(obs[w["head"]]["actions"]):
                events=w["paths"][a];assert sum(Fraction(b["q"]) for b in events)==1
                for b in events:
                    keys=obs[b["tag"]]["actions"];assert set(keys)==set(b["leaves"])
                    for ib,k in enumerate(keys):
                        v=b["leaves"][k];risk=v["direct"] or v["inferred"]
                        outcome=[v["stsr"],risk,v["direct"],v["inferred"],v["abort"] and not risk,0]
                        p=probs[w["head"]][ia]*float(Fraction(b["q"]))*probs[b["tag"]][ib]
                        ref=[x+p*y for x,y in zip(ref,outcome)];rows+=1
            errors.extend(abs(float(m[k].detach())-v) for k,v in zip(FIELDS,ref))
    for domain in sorted({g["domain"] for g in groups}):
        g=next(g for g in groups if g["domain"]==domain and g["split"]=="train")
        w=g["worlds"]["original"];head=w["head"]
        for arm in ("B","C"):
            def f(x,role=None):
                z={t:feature[t]@(x.detach() if role and ((role=="up")!=(t==head)) else x) for t in tags(w)}
                return compute(g,obs,z,arm)[0]
            grad=lambda role:torch.autograd.grad(f(theta,role),theta)[0]
            total,up,down=(grad(role) for role in (None,"up","down"))
            basis=torch.eye(len(theta),dtype=theta.dtype)*1e-5
            fd=torch.stack([(f(theta+d)-f(theta-d))/(2e-5) for d in basis])
            err=(total-fd).abs().max().item();decomp=(total-up-down).abs().max().item()
            assert err<1e-7 and decomp<1e-12 and min(up.norm(),down.norm())>1e-8
            gradients.append(dict(domain=domain,arm=arm,up=up.norm().item(),down=down.norm().item(),finite_difference_error=err,decomposition_error=decomp))
    g=copy.deepcopy(groups[0]);w=g["worlds"]["original"];b=next(iter(w["paths"].values()))[0]
    b["leaves"].pop(next(iter(b["leaves"])))
    z={t:feature[t]@theta for t in tags(w)}
    # Use the non-pair path if removing a paired annotation would also invalidate its conditional target.
    g["matches"]=[];g["hold"]=None
    _,m=compute(g,obs,z)
    assert m["unknown"].item()>0
    for arm in "ABC":
        try:compute(g,obs,z,arm)
        except ValueError as ex:assert "UNKNOWN" in str(ex)
        else:raise AssertionError("Partial label entered training")
    assert max(errors)<1e-12
    dump(HERE/"CPU_CHECK.json",dict(verdict="PASS",worlds=sum(len(g["worlds"]) for g in groups),enumerated_leaves=rows,max_expected_mass_error=max(errors),unknown_fixture_mass=m["unknown"].item(),shared_gradient=gradients))
if __name__=="__main__":main()
