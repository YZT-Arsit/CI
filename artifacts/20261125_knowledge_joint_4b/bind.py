"""Bind a source-train-only subset to the existing native executors and archives."""
import gzip, itertools, json, sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE.parent
sys.path[:0] = [str(ART/"20261125_continuation_4b"), str(ART.parent/"code")]
from prepare import CORE, TABLE, bind_native, env_for, context, pw, cn, strat, js, dump
from results import signed_run
from kci.v9_execution import Execution
import build_negotiation_core as ng

def order(x):
    return pw.h([7, x])

def readlines(p):
    with (gzip.open(p,"rt",encoding="utf8") if p.suffix==".gz" else p.open(encoding="utf8")) as f:
        yield from map(json.loads, f)

def register(ctx, menu):
    actions = sorted(map(cn, menu))
    assert actions and len(actions)==len(set(actions))
    tag = pw.h([ctx, actions])
    obs[tag] = dict(context=ctx, actions=actions)
    return tag

class Mediator(Execution):
    def ask(self, ctx, menu):
        return register(ctx, menu)

def outcome(v):
    return {k:bool(v[k]) for k in ("stsr","direct","inferred","abort")}

def pick(rows, aliases, used, bucket):
    pools = defaultdict(list)
    for r in sorted(rows,key=lambda r:order(r["row_id"])):
        if not used.intersection(aliases(r)):
            pools[bucket(r)].append(r)
    selected=[]
    while len(selected)<16:
        before=len(selected)
        for k in sorted(pools):
            while pools[k]:
                r=pools[k].pop()
                if not used.intersection(aliases(r)):
                    used.update(aliases(r))
                    selected.append(r)
                    break
            if len(selected)==16:break
        if len(selected)==before:raise ValueError("Insufficient distinct source-train groups")
    return selected

def proof(r, specs, sources, raw):
    rid=r["row_id"]
    aa_pair=list(map(cn,r["annotated_first_actions"]))
    win=r["evaluator_only"]["safe_reference"].index(1)
    pair=[aa_pair[win],aa_pair[1-win]]
    worlds={}
    for norm in ("original","authorized"):
        e=env_for(r,specs,sources,norm=="authorized")
        ledger,_=e.prefix()
        ca=context(e,ledger,"A")
        aa=e.messages("source_A")
        if norm=="original":assert ca==r["actor_input"]["actor_context"] and aa==r["actor_input"]["executable_actions"]
        head=register(ca,aa)
        paths={}
        for a in aa:
            ak=cn(a)
            after,_,_=e.deliver(ledger,a)
            cb=context(e,after,"B")
            assert cb["recipient_actual_deliveries"][-1]==a and "authorized_downstream_candidate_catalog" not in cb
            menu=e.messages("source_B")+[dict(kind="SAFE_ABORT",actor="source_B")]
            t=register(cb,menu)
            leaves={}
            for b in menu:
                bk=cn(b)
                result=signed_run(e,ak,bk)
                leaves[bk]=outcome(result)
                old=raw.get((rid,ak,bk))
                if old:
                    pre="authorized_" if norm=="authorized" else ""
                    assert (result["success"], result["risk"])==(old[pre+"task_supported"],old[pre+"ci_violation"])
                    audit["proof_archive_matches"]+=1
            paths[ak]=[dict(tag=t,q="1",event="fixed",leaves=leaves)]
        worlds[norm]=dict(head=head,paths=paths)
    return worlds,pair

def neg(r, env):
    paths={}
    for a in r["executable_actions"]:
        paths[cn(a)]=[dict(tag=b["policy_input"],q=b["probability"],event=b["condition"],
                            leaves={k:outcome(v) for k,v in b["outcomes"].items()})
                      for b in env.negotiation(r,a)]
        audit["neg_verified_events"]+=len(paths[cn(a)])
    chosen,rejected=(cn(r["supervision"][k]) for k in ("chosen_action","rejected_action"))
    return {"original":dict(head=register(r["actor_context"],r["executable_actions"]),paths=paths)},[chosen,rejected]

def targets(w):
    paths=w["paths"]
    assert set(paths)==set(obs[w["head"]]["actions"])
    best={}
    for a,branches in paths.items():
        assert sum(Fraction(b["q"]) for b in branches)==1
        reach=Fraction(0)
        for b in branches:
            assert set(b["leaves"])==set(obs[b["tag"]]["actions"])
            positives=sorted(k for k,v in b["leaves"].items() if v["stsr"])
            reach+=Fraction(b["q"])*bool(positives)
            b["open"]=bool(positives)
            b["accept"]=positives or sorted(k for k,v in b["leaves"].items() if v["abort"] and not(v["direct"] or v["inferred"]))
            # A prior violation can make even an abort unsafe. Such a context has no verified CE target.
        best[a]=reach
    w["feasibility"]={a:float(v) for a,v in best.items()}
    w["accept"]=sorted(a for a,v in best.items() if v==max(best.values()) and v>0)

def match(w,pair):
    left,right=(w["paths"][a] for a in pair)
    result=[]
    for l,r in zip(left,right,strict=True):
        assert l["event"]==r["event"] and l["q"]==r["q"] and l["leaves"].keys()==r["leaves"].keys()
        if l["open"] and not r["open"] and r["accept"]:
            result.append([l["tag"],r["tag"]])
    return result

def irrelevant(w):
    for x,y in itertools.combinations(w["paths"].values(),2):
        for a,b in zip(x,y,strict=True):
            if a["event"]==b["event"] and a["leaves"]==b["leaves"] and a["accept"] and a["tag"]!=b["tag"]:
                return [a["tag"],b["tag"]]
    return None

if __name__=="__main__":
    obs={}; audit=defaultdict(int)
    fresh=js(ART/"20261125_readout_generalization/fresh_freeze.json")
    multi=js(ART/"20261125_multisource_policy/SOURCE_SPLITS.json")
    prior=set(fresh["excluded_groups"])|{s["root"] for s in fresh["selected"]}|set(multi["historical_exclusions"])
    prior.update(x[k] for x in multi["rows"] for k in ("root","source"))
    src=CORE/"D_future_core_multisource_v2/D_future_core.jsonl"
    allspec=list(pw.jl(CORE/"data/PROOFWRITER_CORE_PREREG_V1/PREREG_MINED_SPECS.jsonl"))
    meta={x["spec_id"]:x for x in allspec}
    terms=lambda s:{s["source_group_key"],s["original_source_id"],s["theory_signature"]}
    used=set(prior)
    nr=[r for r in readlines(src) if r["source_dataset"]=="DEAL_OR_NO_DEAL"]
    # Close source/group aliases across the complete source-train index.
    alias_sets=[terms(s) for s in allspec]+[{r["split_group"],r["source_group_key"]} for r in nr]
    while True:
        size=len(used)
        for keys in alias_sets:
            if keys&used:used.update(keys)
        if len(used)==size:break
    pwrows=pick(list(pw.jl(TABLE/"CASES.jsonl")),lambda r:terms(meta[r["row_id"]]),used,strat)
    assert all(meta[r["row_id"]]["split"] in ("train","source_train") for r in pwrows)
    ngrows=pick(nr,lambda r:{r["split_group"],r["source_group_key"]},used,lambda r:r["core_mechanism"])
    assert all(r["split"]=="source_train" for r in ngrows)
    ids={r["row_id"] for r in pwrows}
    raw={(x["row_id"],cn(x["first_action"]),cn(x["next_action"])):x["evaluator_target"]
         for x in readlines(TABLE/"EVAL_ONLY_CELLS.jsonl") if x["row_id"] in ids}
    specs,sources=bind_native(pwrows)
    arc=CORE/"data/NEGOTIATION_CORE_V2/EVAL_ONLY_ALL_NATIVE_BRANCHES.jsonl.gz"
    assert pw.sh(arc)==js(arc.parent/"MANIFEST.json")["witness_sha256"]
    env=Mediator.__new__(Mediator);env.ng=ng
    ids={r["row_id"] for r in ngrows};env.neg={x["id"]:x for x in readlines(arc) if x["id"] in ids}
    groups=[]
    for domain,rows in (("ProofWriter",pwrows),("DealOrNoDeal",ngrows)):
        strata=defaultdict(list)
        for r in rows:
            strata[strat(r) if domain=="ProofWriter" else r["core_mechanism"]].append(r)
        ordered=[r for k in sorted(strata) for r in strata[k]]
        train_ids={r["row_id"] for r in ordered[::2]}
        for i,r in enumerate(rows):
            worlds,pair=proof(r,specs,sources,raw) if domain=="ProofWriter" else neg(r,env)
            for w in worlds.values():targets(w)
            w=worlds["original"];pairs=match(w,pair)
            source=meta[r["row_id"]]["original_source_id"] if domain=="ProofWriter" else r["source_group_key"]
            groups.append(dict(domain=domain,root=r["split_group"],source=source,row=r["row_id"],split="train" if r["row_id"] in train_ids else "dev",
                               worlds=worlds,pair=pair,matches=pairs,hold=irrelevant(w)))
    def tags(g):
        return {t for w in g["worlds"].values() for t in [w["head"]]+[b["tag"] for bs in w["paths"].values() for b in bs]}
    for key in ("source","root"):
        assert len({g[key] for g in groups})==len(groups)
    assert not set.union(*(tags(g) for g in groups if g["split"]=="train"))&set.union(*(tags(g) for g in groups if g["split"]=="dev"))
    for name,data in (("groups",groups),("observations",obs)):
        with gzip.open(HERE/(name+".json.gz"),"wt",encoding="utf8") as f:json.dump(data,f,separators=(",",":"))
    dump(HERE/"binding.json",dict(audit=dict(audit),inputs={str(p):pw.sh(p) for p in [src,arc,TABLE/"CASES.jsonl",TABLE/"EVAL_ONLY_CELLS.jsonl",ART/"20261125_readout_generalization/fresh_freeze.json",ART/"20261125_multisource_policy/SOURCE_SPLITS.json",Path(__file__).resolve()]},selected=[{k:g[k] for k in ("domain","row","root","source","split")} for g in groups],observations=len(obs),exposure="Source train only. Outside recorded exclusions and later multisource and fresh panel. Independent blindness unverified."))
    print("bound",len(groups),len(obs),dict(audit),flush=True)
