"""Controlled native-history comparisons for KCI-Align v0.8."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction as F
import gzip
import json
from pathlib import Path

from .paper_alignment import actor, prompt
from .paper_run import unpack, dump
from .paper_worlds import PhaseWorld, canon, filehash, key


def stream(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f]


def order(v):
    risk, success, cost_mass = v
    return -risk, success, -cost_mass / success if success else 0


def values(v):
    return [str(v[0]), str(v[1]), str(v[2]/v[1]) if v[1] else None]


class Planner:
    """Exact bounded native Bellman recursion. Cost is incremental from the queried state."""
    def __init__(self, world):
        self.w = world
        self.memo = {}

    def value(self, s, left, mode):
        done, success, _, _ = self.w.terminal(s)
        if done or left == 0:
            return F(0), F(done and success), F(0)
        w = self.w
        if isinstance(w, PhaseWorld):
            state = key(asdict(s))
        else:
            # Within a fixed native world each stage transmits an immutable projection.
            # Retry probabilities are constant. Both the reference and legal transitions
            # use the failure count only through zero, one, and at least two.
            state = (s.stage, tuple(sorted(w.env._recipient_views(s))), tuple(sorted(w.env._broker_views(s))),
                     s.authorized_relation, s.attempted_consent, min(s.private_failures,2),
                     s.business_result, s.last_failure)
        tag = state, left, mode
        if tag not in self.memo:
            ob = w.observe(s)
            actions = [w.reference(ob)] if mode == "reference" else ob["legal_actions"]
            candidates = [self.action(s,a,left,mode) for a in actions]
            self.memo[tag] = max((v for v in candidates if v is not None), key=order)
        return self.memo[tag]

    def action(self, s, a, left, mode):
        out = [F(0), F(0), F(0)]
        branches = self.w.branches(s,a)
        assert sum(p for p,_,_ in branches)==1
        cost = self.w.terminal(s)[2]
        for p,n,audit in branches:
            if audit["verdict"]=="VIOLATION":
                if mode=="optimal":
                    return None
                out[0]+=p
                continue
            if audit["verdict"]!="VALID":
                raise ValueError("Unresolved transition in finite calibration")
            risk, success, cm = self.value(n,left-1,mode)
            out[0]+=p*risk
            out[1]+=p*success
            out[2]+=p*(cm+success*F(str(self.w.terminal(n)[2]-cost)))
        return tuple(out)


def frozen_audit(run):
    prefs=stream(run/"preferences.jsonl.gz")
    evidence={(r["world"],r["state"],r["remaining"]):r for r in stream(run/"fit_evidence.jsonl.gz")}
    pool, drawn, roots = Counter(), Counter(), Counter()
    tags={}
    risk_diffs=0
    for r in prefs:
        label=r["source"]
        if label=="future":
            rs={canon(c["action"]):c["reference"] for c in evidence[r["world"],r["state"],r["remaining"]]["candidates"] if "reference" in c}
            good,bad=rs[canon(r["chosen"])],rs[canon(r["rejected"])]
            neg=r["rejected"]
            if good["any_ci"]!=bad["any_ci"]:risk_diffs+=1
            if "ABORT" in canon(neg):label="abort"
            elif isinstance(neg,dict) and "answer" in neg:label="terminal_answer"
            elif good["stsr"]!=bad["stsr"]:label="other_success"
            else:
                assert good["cost_given_safe_success"]!=bad["cost_given_safe_success"]
                label="cost"
        tags[key(r)]=(label,r["root"])
        pool[label]+=1
    for step in json.loads((run/"optimizer.json").read_text())["steps"]:
        for i in step["row_ids"]:
            label,root=tags[i]
            drawn[label]+=1
            roots[root]+=1
    return dict(pool=pool,drawn=drawn,draw_roots=roots,future_risk_differences=risk_diffs)


def advance(w,s,a):
    b=w.branches(s,a)
    assert len(b)==1 and b[0][0]==1 and b[0][2]["verdict"]=="VALID"
    return b[0][1]


def route(w,s,kind):
    return next(a for a in w.observe(s)["legal_actions"] if a["kind"]==kind)


def phase_prefix(w, risk):
    s=advance(w,w.initial(),"A_READ_VERIFIED_PACKAGES")
    risky=w.env.packet_by_kind["risky"]
    packet=risky if risk else next(x for x in s.packages if x!=risky)
    s=advance(w,s,"A_SEND_"+packet)
    while s.stage=="pre":
        s=advance(w,s,next(a for a in w.observe(s)["legal_actions"] if a.startswith("DO:")))
    assert s.stage=="preflight"
    return s


def history_probe(w,s,raw):
    audit=w.branches(s,raw)[0][2]
    return dict(world=w.id,root=w.root,prompt=prompt(w.observe(s)),raw=raw,
                verdict=audit["verdict"],direct=audit["direct"],inferred=audit["inferred"])


def compare_world(w):
    phase=isinstance(w,PhaseWorld)
    family="phase2" if phase else w.env.family
    planner=Planner(w)
    probes={}
    contrasts={}
    for history in ("missing","present"):
        seen=history=="present"
        if phase:
            s=phase_prefix(w,seen)
            options=["OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"]
            left=24
            def at_release(start):
                n=advance(w,start,"OPEN_PRIVATE_CHANNEL")
                return advance(w,n,next(a for a in w.observe(n)["legal_actions"] if a.startswith("READ:")))
            probe_state=at_release(s)
            raw="SEND_RAW_TARGET"
        else:
            s=w.initial()
            s=advance(w,s,route(w,s,w.mod.PUB if seen else w.mod.PRIVATE))
            if family=="TWO_HOP_JOIN":
                options=[route(w,s,k) for k in (w.mod.PUB,w.mod.PRIVATE)]
                left=9
                probe_state=advance(w,s,route(w,s,w.mod.PUB if seen else w.mod.PRIVATE))
                if not seen:
                    irrelevant=advance(w,s,route(w,s,w.mod.PUB))
                    probes["irrelevant"]=history_probe(w,irrelevant,route(w,irrelevant,w.mod.PUB))
            else:
                options=[]
                probe_state=s
            raw=route(w,probe_state,w.mod.PUB)
        probes[history]=history_probe(w,probe_state,raw)
        if options:
            ob=w.observe(s)
            q={}
            for mode in ("reference","optimal"):
                vs=[planner.action(s,a,left,mode) for a in options]
                assert all(v is not None for v in vs)
                q[mode]=[values(v) for v in vs]
            contrasts[history]=dict(prompt=prompt(ob),actor=actor(ob),options=options,q=q,
                                    starting_cost=w.terminal(s)[2],state=key(asdict(s)))
    return dict(root=w.root,world=w.id,family=family,probes=probes,history=contrasts)


def diffs(row,mode):
    vals=row["q"][mode]
    return tuple(F(a)-F(b) if a is not None and b is not None else None for a,b in zip(*vals))


def direction(row,mode):
    v=diffs(row,mode)
    for d in (-v[0],v[1],-v[2] if v[2] is not None else 0):
        if d:return 1 if d>0 else -1
    return 0


def audit(packet, pilot, out):
    out.mkdir(parents=True,exist_ok=True)
    worlds=list(unpack(packet))
    retired={w.root for _,w in unpack(pilot/"packet")}
    dump(out/"prior_exposure.json",frozen_audit(pilot/"run02"))
    groups=defaultdict(dict)
    rows=[]
    for split,w in worlds:
        rec=compare_world(w)
        rec.update(split=split,retired=w.root in retired)
        if isinstance(w,PhaseWorld):
            group=(w.root,w.env.swap,w.env.reorder,w.env.variant)
            auth=w.env.authorized
        else:
            group=(w.root,w.env.mechanism_id)
            auth=w.env.preauthorized
        groups[group][auth]=rec
        rows.append(rec)
    core=[];summary=Counter();panels=[]
    for group,conditions in groups.items():
        x=conditions[False]
        auth=conditions.get(True)
        ps=x["probes"]
        flip=ps["missing"]["verdict"]=="VALID" and ps["present"]["inferred"]
        summary["history_flip_groups"]+=int(flip)
        if auth:
            ap=auth["probes"]["present"]
            if x["family"]=="phase2":
                hold=groups[(x["root"],False,False,"hold")][False]["probes"]["missing"]
            else:hold=ps.get("irrelevant")
            if hold and flip and ap["verdict"]==hold["verdict"]=="VALID":
                panels.append(dict(root=x["root"],split=x["split"],retired=x["retired"],group=list(group),
                                   cases=dict(legal=ps["missing"],danger=ps["present"],authorized=ap,irrelevant=hold)))
        if not x["history"] or not auth:continue
        for side,case in x["history"].items():
            assert case["options"]==x["history"]["missing"]["options"] and case["actor"]==x["history"]["missing"]["actor"]
        ds={}
        for mode in ("reference","optimal"):
            def dd(con):
                a=diffs(con["history"]["present"],mode)
                b=diffs(con["history"]["missing"],mode)
                return [v-u if v is not None and u is not None else None for v,u in zip(a,b)]
            ds[mode]=dd(x)
            if any(dd(auth)):
                raise ValueError("Authorization control retains history-dependent route value")
        summary["matched_four_cells"]+=1
        if not flip or not any(ds["reference"]) or not any(ds["optimal"]):continue
        for side,case in x["history"].items():
            ref=direction(case,"reference");opt=direction(case,"optimal")
            summary["calibrated_cells"]+=1
            summary["matching_ranks"]+=int(ref==opt)
            if ref!=opt or not opt:continue
            pick=0 if opt>0 else 1
            core.append(dict(root=x["root"],world=x["world"],group=list(group),split=x["split"],retired=x["retired"],
                history=side,prompt=case["prompt"],chosen=case["options"][pick],rejected=case["options"][1-pick],
                q=case["q"],did={k:[str(v) if v is not None else None for v in vs] for k,vs in ds.items()}))
    for name,data in (("native",rows),("panels",panels),("core",core)):
        with gzip.open(out/(name+".jsonl.gz"),"wt",encoding="utf8") as f:
            for r in data:f.write(canon(r)+"\n")
    dump(out/"audit.json",dict(packet=filehash(packet/"manifest.json"),roots=len({w.root for _,w in worlds}),
        worlds=len(worlds),groups=len(groups),counts=summary,rows=len(core),
        fit_core_roots=len({r["root"] for r in core if r["split"]=="fit"}),
        fresh_core_roots=len({r["root"] for r in core if r["split"]=="validation" and not r["retired"]}),
        four_case_groups=len(panels)))
    return rows,core,panels


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    for k in ("packet","pilot","output"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();audit(a.packet,a.pilot,a.output)

