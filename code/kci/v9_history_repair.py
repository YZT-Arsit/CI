"""Acquire true DPR failures and match the identical B actions across delivery histories."""
import json
from pathlib import Path
from v9_schedule import rng


def context(prompt):
    return json.loads(prompt.split("\nExecutable actions:\n")[0].split("\n",1)[1])


def cases(graphs, prompts, baseline):
    model={r["id"]:r for r in baseline["root_results"]}
    evidence={}
    for g in graphs:
        r=model[g["id"]]; a=r["first"]
        b=g["paths"][a][0]; t=b["policy_input"]
        other=g["paths"][g["rejected"]][0];u=other["policy_input"]
        reached=r["events"][a][0]; y=reached["action"]
        assert reached["input"]==t and b["outcomes"][y]["stsr"]==reached["stsr"]
        safe=[x for x,v in b["outcomes"].items() if v["stsr"]]
        pairs=[]
        if g["valid_first"][a] and not reached["stsr"] and safe:
            left,right=context(prompts[t]["prompt"]),context(prompts[u]["prompt"])
            lh=left.pop("recipient_actual_deliveries");rh=right.pop("recipient_actual_deliveries")
            assert left==right and set(b["outcomes"])==set(other["outcomes"])
            blocked=other["outcomes"]
            if lh!=rh and not (blocked[y]["direct"] or blocked[y]["inferred"]):
                pairs=[x for x in safe if blocked[x]["success"] and (blocked[x]["direct"] or blocked[x]["inferred"])]
        evidence[g["id"]]=dict(root=g["source_root"],id=g["id"],split=g["split"],first=a,input=t,hazard_input=u,
            failure=y,original=reached,legal_first=g["valid_first"][a],safe_alternatives=safe,matched=pairs)
    return evidence


def make_schedule(gs, evidence, seed):
    by={g["id"]:g for g in gs}
    pool=[v for v in evidence.values() if v["matched"]]
    # Membership is seed independent; only draws and exposure order vary across seeds.
    pool=sorted(pool,key=lambda v:rng(7,"membership",v["id"]).random())[:24]
    arms={a:[] for a in ("ordinary","dpr","repair_ce","generic_ci","paired_ci")}
    for epoch in range(2):
        ordered=pool.copy();rng(seed,epoch,"order").shuffle(ordered)
        for v in ordered:
            g=by[v["id"]]; i=v["id"]; y=v["failure"]; t=v["input"]; u=v["hazard_input"]
            def pick(k,options):return rng(seed,epoch,i,k).choice(options)
            x=pick("repair",v["matched"])
            root_pairs=[];demos=[]
            for first,branches in g["paths"].items():
                if not g["valid_first"][first]:continue
                for b in branches:
                    z=b["policy_input"];s=b["outcomes"]
                    good=[a for a,o in s.items() if o["stsr"]]
                    demos.extend([[z,a]] for a in good)
                    root_pairs.extend(([[z,a]],[[z,bad]]) for a in good for bad,o in s.items() if not o["stsr"])
            common=pick("common",demos)
            positive,negative=pick("ordinary",root_pairs)
            hazard=g["paths"][g["rejected"]][0]["outcomes"]
            ci=pick("generic",[(a,b) for a,o in hazard.items() if not(o["inferred"] or o["direct"])
                               for b,n in hazard.items() if n["direct"] or n["inferred"]])
            for arm in arms:
                p,n=([[t,x]],[[t,y]])
                if arm=="ordinary":p,n=positive,negative
                if arm=="repair_ce":n=[]
                c=common;cn=[]
                if arm in ("generic_ci","paired_ci"):
                    pos,neg=ci if arm=="generic_ci" else (y,x)
                    c,cn=[[u,pos]],[[u,neg]]
                arms[arm].append(dict(id=i,root=v["root"],first=v["first"],common=c,common_neg=cn,pos=p,neg=n))
    return arms

