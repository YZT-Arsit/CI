"""Rebind existing preferences, regress native route constraints, and freeze source-disjoint panels."""
import gzip
import itertools
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

P=Path(__file__).resolve().parent
ART=P.parent
sys.path.insert(0,str(ART.parent/"code"))
from kci.paper_run import unpack
from kci.paper_alignment import prompt
from kci.paper_worlds import RelWorld, canon, key, filehash
from kci.mechanism_v08 import Planner, order

PACK=ART/"20261124_paper_native_gpu_18828/full_train_packet"
REL=ART/"20261125_unified_roles_v1"
PUB="SEND_PUBLIC_PROJECTION"
PRIV="SEND_BROKER_PRIVATE"
def dump(n,x):
    with gzip.open(P/(n+".json.gz"),"wt",encoding="utf8") as f:json.dump(x,f,ensure_ascii=False,separators=(",",":"))
def readlines(path):
    with gzip.open(path,"rt",encoding="utf8") as f:
        yield from map(json.loads,f)
def context(w,s):
    ob=w.observe(s); p=prompt(ob)
    return p, sorted(map(canon,ob["legal_actions"]))
def bind_tag(p,menu):
    ob=json.loads(p)
    for k in ("legal_actions","oracle_aggregate_result_NOT_given","version"):ob.pop(k,None)
    for i,ev in enumerate(ob.get("actual_delivery_receipt_metadata",[]),1):
        ev["event_id"]="event_"+str(i)
    return key([ob,menu])
def select(w,s,kind):
    return next(a for a in w.observe(s)["legal_actions"] if a["kind"]==kind)
def delivery(w,s,kind):
    a=select(w,s,kind)
    options=[(p,t,u) for p,t,u in w.branches(s,a) if t.stage==s.stage+1]
    return options[0] if options else (0,None,None)

def matrix(w,s):
    env=w.env
    stages=list(env.stages[s.stage:])
    if not stages:return []
    origin=set(env._recipient_views(s)) & set(env.stages)
    supports=[]
    for n in range(1,len(env.stages)+1):
        for subset in itertools.combinations(env.stages,n):
            if any(v<=set(subset) for v in supports):continue
            visible={k:env.model["source_views"][k] for k in subset}
            if w.mod.join_from_views(visible,env.stages):supports.append(set(subset))
    assert supports==[set(env.stages)]
    assert all(not f<=origin for f in supports)
    entries=[]
    for route in itertools.product((PUB,PRIV),repeat=len(stages)):
        t=s;received=set();bad=False;mass=Fraction(1)
        for stage,kind in zip(stages,route,strict=True):
            prob,child,audit=delivery(w,t,kind)
            if not prob:break
            mass*=prob;bad |= audit["verdict"]=="VIOLATION"
            assert audit["verdict"] in ("VALID","VIOLATION") and not audit["direct"]
            received |= {ev["source_stage"] for ev in audit["events"] if ev["recipient"]==w.mod.RECIPIENT and ev["status"]=="DELIVERED"}
            t=child
        else:
            compute=select(w,t,w.mod.COMPUTE);[(one,t,au)]=w.branches(t,compute)
            receipt=w.observe(t)["actually_received_aggregate_receipt"]
            final=next(a for a in w.observe(t)["legal_actions"] if a["kind"]==w.mod.FINAL and a["answer"]==receipt["threshold_met"])
            [(_,end,au2)]=w.branches(t,final)
            assert one==1 and au["verdict"]==au2["verdict"]=="VALID" and w.terminal(end)[:2]==(True,True)
            residual=[sorted(f-received) for f in supports] if not s.authorized_relation else []
            y=not any(set(r)<=origin for r in residual)
            assert y==(not bad)==(not end.any_violation)
            steps=[dict(source_stage=k,kind=a) for k,a in zip(stages,route)]
            steps.extend([{"kind":w.mod.COMPUTE},{"kind":w.mod.FINAL,"answer":"copy signed aggregate receipt"}])
            entries.append(dict(path=steps,y=y,residual=residual,K=sorted(origin),D=sorted(received),mass=str(mass),cost=end.cost-s.cost))
    return entries

def main():
    split=json.loads((REL/"SPLITS.json").read_text(encoding="utf8"))
    alias={a:k for k,v in split["components"].items() for a in v}
    part={g:k for k,gs in split["partitions"].items() for g in gs}
    archive=defaultdict(lambda:defaultdict(list))
    for r in readlines(REL/"train_dpo.jsonl.gz"):
        if r["domain"]!="MULTIAGENTPRIVACY" or r["actor_context"].get("family")!="DIRECT_JOIN":continue
        if not set(r["tiers"]) & {"core_reference","train_actual_failure_dpr"}:continue
        menu=sorted(map(canon,r["executable_actions"]))
        archive[r["source"].split(":",1)[1]][bind_tag(canon(r["actor_context"]),menu)].append(r)
    worlds=list((sp,w) for sp,w in unpack(PACK) if isinstance(w,RelWorld))
    training=[];chosen_groups=set(); stats=Counter(); pools={}
    for sp,w in worlds:
        group=alias["MULTIAGENTPRIVACY:"+w.root]
        if w.root not in archive or sp!="fit" or part.get(group)!="train" or w.env.family!="DIRECT_JOIN":continue
        if w.env.preauthorized:continue
        pl=Planner(w); start=w.initial()
        # States with actually reachable signed receipts; pair each indexed state with both existing teachers.
        states=[start]+[delivery(w,start,k)[1] for k in (PUB,PRIV)]
        for s in states:
            if s is None:continue
            p,menu=context(w,s)
            bound=archive[w.root].get(bind_tag(p,menu),[])
            if not bound:continue
            vs=[pl.action(s,json.loads(a),10-s.steps,"optimal") for a in menu]
            best=max(order(v) for v in vs if v is not None)
            good=[j for j,v in enumerate(vs) if v is not None and order(v)==best]
            cells=matrix(w,s)
            if not cells:continue
            for r in bound:
                rec=dict(root=w.root,group=group,world=w.id,record=r["record_id"],tiers=r["tiers"],p=p,menu=menu,
                         good=good,pos=[menu.index(a) for a in r["accepted"]],neg=[menu.index(a) for a in r["rejected"]],
                         stage=w.env._stage(s),cells=cells)
                pools.setdefault(group,[]).append(rec)
    groups=sorted(pools,key=lambda g:key(["fixed-training-v1",g]))[:24]
    for group in groups:
        training.extend(pools[group]);chosen_groups.add(group)
    assert any("core_reference" in r["tiers"] for r in training) and any("train_actual_failure_dpr" in r["tiers"] for r in training)
    # Bind both actual delivery histories and their authorized counterparts for every training source.
    # Reuse archived records whenever present. At uncovered states re-evaluate the original fixed
    # reference with the same R,S,conditional-cost ordering and explicitly record this provenance.
    seen={(r["world"],r["p"]) for r in training}
    train_roots={r["root"] for r in training}
    for sp,w in worlds:
        if w.root not in train_roots or w.env.family!="DIRECT_JOIN" or sp!="fit":continue
        group=alias["MULTIAGENTPRIVACY:"+w.root]
        if group not in chosen_groups:continue
        pl=Planner(w);start=w.initial()
        for s in [start]+[delivery(w,start,k)[1] for k in (PUB,PRIV)]:
            if s is None:continue
            p,menu=context(w,s)
            if (w.id,p) in seen:continue
            vals=[pl.action(s,json.loads(a),10-s.steps,"optimal") for a in menu]
            best=max(order(v) for v in vals if v is not None)
            refs=[order(pl.action(s,json.loads(a),10-s.steps,"reference")) for a in menu]
            preferred=[j for j,v in enumerate(refs) if v==max(refs)]
            # Among lower ranked actions use the hardest distinct reference rank, retain ties.
            lower=[v for v in refs if v<max(refs)]
            if not lower:continue
            rejected=[j for j,v in enumerate(refs) if v==max(lower)]
            training.append(dict(group=group,root=w.root,world=w.id,record=key([w.id,p,"replayed_CCE"]),tiers=["replayed_fixed_reference_CCE"],
                p=p,menu=menu,good=[j for j,v in enumerate(vals) if v is not None and order(v)==best],pos=preferred,neg=rejected,
                stage=w.env._stage(s),cells=matrix(w,s)))
            seen.add((w.id,p))
    # Older validation partition has independently assigned source roots; also retain unified-release validation and development.
    candidates=defaultdict(list)
    for sp,w in worlds:
        group=alias["MULTIAGENTPRIVACY:"+w.root]
        if w.env.preauthorized or group in chosen_groups:continue
        if sp!="validation" and part.get(group) not in ("validation","development"):continue
        candidates[w.env.family].append((group,w.root,w.id))
    selected={}
    for fam,limit in [("DIRECT_JOIN",8),("TWO_HOP_JOIN",24)]:
        for group,root,world in sorted(candidates[fam],key=lambda t:key(["fixed-dev-v1",t])) :
            if group in selected:continue
            if sum(f==fam for f,_,_ in selected.values())>=limit:break
            selected[group]=(fam,root,world)
    # Save test INPUTS separately from tables. Evaluation enumerates native actions with a fixed original seed.
    observations={}; games=[]; all_cells=[]; nodes=[]
    def register(w,s):
        p,menu=context(w,s);tag=key([p,menu]);observations[tag]=dict(p=p,menu=menu);return tag
    def tree(w,s):
        at=len(nodes);nodes.append(None)
        done,succ,cost,abort=w.terminal(s)
        if done or s.steps>=10 or s.any_violation:
            nodes[at]=dict(stsr=bool(done and succ and not s.any_violation),risk=s.any_violation,
                           abort=abort,cost=cost,timeout=bool(not done and not s.any_violation),steps=s.steps)
        else:
            tag=register(w,s);nexts=[]
            for val in observations[tag]["menu"]:
                t,au=w.env.transition(s,json.loads(val));assert au["verdict"] in ("VALID","VIOLATION")
                nexts.append(tree(w,t))
            nodes[at]=dict(tag=tag,children=nexts)
        return at
    fit_eval=set(sorted(chosen_groups,key=lambda g:key(["fit-eval",g]))[:4])
    fit_ids={r["world"]:r["group"] for r in training if r["group"] in fit_eval}
    for sp,w in worlds:
        group=alias["MULTIAGENTPRIVACY:"+w.root]
        ident=selected.get(group)
        if w.id not in fit_ids and (not ident or ident[:2]!=(w.env.family,w.root)):continue
        start=w.initial();pl=Planner(w)
        game=dict(root=w.root,group=group,world=w.id,family=w.env.family,role="train_fit" if group in fit_eval else "development",
                  release_split=part[group],authorized=w.env.preauthorized,start=tree(w,start),sides=[])
        for first in (PRIV,PUB):
            _,s,au=delivery(w,start,first);assert au["verdict"]=="VALID"
            tag=register(w,s);menu=observations[tag]["menu"]; options=[select(w,s,k) for k in (PUB,PRIV)]
            vs=[pl.action(s,a,10-s.steps,"optimal") for a in options]
            safe_now=[all(v["verdict"]=="VALID" for _,_,v in w.branches(s,a)) for a in options]
            opt=None if any(v is None for v in vs) or order(vs[0])==order(vs[1]) else int(order(vs[1])>order(vs[0]))
            game["sides"].append(dict(tag=tag,choices=[menu.index(canon(a)) for a in options],opt=opt,safe_now=safe_now,
                                      values=[None if v is None else list(map(str,v)) for v in vs]))
            for c in matrix(w,s):all_cells.append(dict(group=group,world=w.id,family=w.env.family,auth=w.env.preauthorized,
                                                        role=game["role"],history=first,tag=tag,**c))
        for c in matrix(w,start):all_cells.append(dict(group=group,world=w.id,family=w.env.family,auth=w.env.preauthorized,
                                                     role=game["role"],history="initial",tag=register(w,start),**c))
        games.append(game)
    assert set(selected).isdisjoint(chosen_groups)
    assert len({r["group"] for r in training})>=8 and selected and all(g["sides"] for g in games)
    for n,ob in [("training",training),("observations",observations),("graphs",dict(nodes=nodes,games=games)),("matrix",all_cells)]:dump(n,ob)
    report=dict(train_groups=groups,native_train_states=len({r["p"] for r in training}),training_records=len(training),
                archived_tiers=dict(Counter(y for r in training for y in r["tiers"])),dev_groups=selected,
                worlds=len(games),native_nodes=len(nodes),observations=len(observations),
                train_matrix_lengths=dict(Counter(len(c["path"]) for r in training for c in r["cells"])),
                eval_matrix_lengths=dict(Counter(len(c["path"]) for c in all_cells)),matrix_cells=len(all_cells),
                reversals=[g["group"] for g in games if g["role"]=="development" and not g["authorized"] and
                     all(all(s["safe_now"]) for s in g["sides"]) and {s["opt"] for s in g["sides"]}=={0,1}],
                source_packet_hash=filehash(PACK/"packet.json.gz"),data_hash=filehash(REL/"train_dpo.jsonl.gz"))
    (P/"BINDING.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print({k:v if k not in ("train_groups","dev_groups","reversals") else len(v) for k,v in report.items()},flush=True)
if __name__=="__main__":main()
