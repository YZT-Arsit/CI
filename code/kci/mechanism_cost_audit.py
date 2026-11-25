"""Reconstruct the 117 historic cost contrasts under actual alternative lawful prefixes."""
from collections import Counter
from pathlib import Path
import argparse
import json
from .mechanism_v08 import Planner, stream, values
from .paper_alignment import Alignment
from .paper_run import unpack,dump
from .paper_worlds import PhaseWorld,RelWorld,canon,key


def replay(w,path,swap):
    s=w.initial()
    changed=False
    for h in path:
        ob=w.observe(s)
        a=h["action"]
        if isinstance(w,PhaseWorld) and ob["stage"]=="A_SEND" and swap:
            a=next(v for v in ob["legal_actions"] if v.startswith("A_SEND_") and v!=a)
            changed=True
        elif not isinstance(w,PhaseWorld) and ob["role_stage"]=="public" and swap:
            if a["kind"] not in (w.mod.PRIVATE,w.mod.PUB):return None,False
            a=next(v for v in ob["legal_actions"] if v["kind"] in (w.mod.PRIVATE,w.mod.PUB) and v!=a)
            changed=True
        if a not in ob["legal_actions"]:return None,changed
        b=w.branches(s,a)
        if len(b)>1:
            # Match the realized Bernoulli event, including consent grant/denial.
            def signature(audit):
                return [(e.get("route"),e.get("status"),e.get("content",{}).get("granted")
                         if isinstance(e.get("content"),dict) else None) for e in audit["events"]]
            b=[v for v in b if signature(v[2])==signature(h["audit"])]
        if len(b)!=1 or b[0][2]["verdict"]!="VALID":return None,changed
        s=b[0][1]
    return s,changed


def run(pilot,out):
    run=pilot/"run02"
    worlds={w.id:w for _,w in unpack(pilot/"packet")}
    by={(r["world"],r["state"],r["remaining"]):r for r in stream(run/"fit_evidence.jsonl.gz")}
    paths={}
    for t in stream(run/"fit_traces.jsonl.gz"):
        if "path" not in t:continue
        for i,h in enumerate(t["path"]):
            paths[t["initial_world"],h["state"]]=t["path"][:i]
    details=[]
    for r in stream(run/"preferences.jsonl.gz"):
        if r["source"]!="future":continue
        wid,sid,left=r["world"],r["state"],r["remaining"]
        ev=by[wid,sid,left]
        candidates={canon(c["action"]):c["reference"] for c in ev["candidates"] if "reference" in c}
        pair=[r["chosen"],r["rejected"]]
        a,b=[candidates[canon(c)] for c in pair]
        if a["any_ci"]!=b["any_ci"] or a["stsr"]!=b["stsr"]:continue
        assert a["cost_given_safe_success"]<b["cost_given_safe_success"]
        w=worlds[wid]
        prefix=paths[wid,sid]
        record=dict(root=w.root,world=wid,state=sid,actions=pair,old_cost=[a["cost_given_safe_success"],b["cost_given_safe_success"]])
        s,_=replay(w,prefix,False)
        assert Alignment(w,w.reference,"check").sid(s)==sid
        other,changed=replay(w,prefix,True)
        if not changed:label="no_prior_intervention"
        elif other is None:label="prefix_ceases_to_be_lawful_or_executable"
        elif w.env.authorized if isinstance(w,PhaseWorld) else s.authorized_relation:label="already_authorized"
        else:
            def gaps(env,st):
                ob=env.observe(st)
                if any(act not in ob["legal_actions"] for act in pair):return None
                if any(v["verdict"]!="VALID" for act in pair for _,_,v in env.branches(st,act)):return None
                planner=Planner(env)
                qs=[planner.action(st,act,left,"reference") for act in pair]
                gap=[qs[0][j]-qs[1][j] for j in (0,1)]
                cost=[v[2]/v[1] if v[1] else None for v in qs]
                return tuple(gap+[cost[0]-cost[1] if None not in cost else None])
            original=gaps(w,s);counter=gaps(w,other)
            if counter is None:label="immediate_menu_or_legality_changed"
            elif original==counter:label="no_history_interaction"
            else:
                if isinstance(w,PhaseWorld):
                    auth=PhaseWorld(w.env.spec,True,w.env.reorder,w.env.swap,w.env.variant)
                else:auth=RelWorld(w.mod,w.env.spec,w.env.source,True)
                ordinary,_=replay(auth,prefix,False);alternative,_=replay(auth,prefix,True)
                nulls=[gaps(auth,st) if st is not None else None for st in (ordinary,alternative)]
                label="history_effect_removed_by_authorization" if None not in nulls and nulls[0]==nulls[1] else "authorization_control_unresolved"
                record["authorized_gaps"]=[[str(x) for x in v] if v else None for v in nulls]
            record["matched_gaps"]=[[str(x) for x in v] if v else None for v in (original,counter)]
        details.append(dict(**record,label=label))
    dump(out,dict(counts=Counter(r["label"] for r in details),rows=details))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("pilot",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.pilot,a.output)

