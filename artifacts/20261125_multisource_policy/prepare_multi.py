"""Bind separately typed, source-train-only supervision to role-visible actions."""
import gzip,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
OUT=Path(__file__).resolve().parent;ART=OUT.parent
BASE=Path("D:/research/kci_supervision_alignment_20261123")
CORE=BASE/"d_future_core_20261124";V9=BASE/"deliverables/KCI_VALIDATED_V9_INTERNAL"
OLD=ART/"20261125_continuation_4b";NEW=ART/"20261125_readout_generalization"
def cn(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
def key(x):return hashlib.sha256(cn(x).encode()).hexdigest()
def rd(p):return json.loads(p.read_text(encoding="utf8"))
def rows(p):
    with p.open(encoding="utf8") as f:return [json.loads(x) for x in f if x.strip()]
def rgz(p):return json.load(gzip.open(p,"rt",encoding="utf8"))
def gz(p,x):
    with gzip.open(p,"wt",encoding="utf8",compresslevel=3) as f:json.dump(x,f,ensure_ascii=False,separators=(",",":"))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rr(seq,k):
    buckets=defaultdict(list)
    for r in sorted(seq,key=lambda x:key(["multi-v1",x["row_id"]])):buckets[r[k]].append(r)
    while any(buckets.values()):
        for b in sorted(buckets):
            if buckets[b]:yield buckets[b].pop()
def inputs():
    manifest={};allrows=[];counts={}
    def take(file,channel):
        rs=rows(file);manifest[str(file)]=sha(file)
        counts[channel]={"available_rows":len(rs),"source_roots":len({r["source_group_key"] for r in rs}),"groups":len({r.get("split_group",r["source_group_key"]) for r in rs})}
        for r in rs:
            r["tier"]=channel;r["group"]=r.get("split_group",r["source_group_key"])
            r["mechanism"]=r.get("core_mechanism",r.get("stage",channel));r["domain"]=r["source_dataset"] if channel!="TASK_ONLY" else "TASK_ONLY"
            assert r["split"] in ("train","source_train")
            if channel=="controls_only":r["eval_only"]=True
            allrows.append(r)
    f=CORE/"D_future_core_multisource_v2/D_future_core.jsonl";take(f,"core_reference")
    for f in sorted((V9/"typed").glob("*.jsonl")):
        if f.stem=="diagnostic_reference_only":continue
        take(f,f.stem)
    take(CORE/"BUSINESS_NORM_SUPPLEMENT_V1/task_support/TRAIN_SOURCE_GROUNDED_TASK_ONLY.jsonl","TASK_ONLY")
    return allrows,manifest,counts
def bound_exposure(rs):
    banned={"PHASE2:P207","PHASE2:P211","PHASE2:P212"};old=set()
    for s in ("20261124_v9_training/freeze.json","20261124_v9_history_repair/freeze.json","20261125_icml_v02/ROOT_MANIFEST.json","20261125_icml_v02/B1/ROOT_MANIFEST.json"):
        m=rd(ART/s)
        for g in m.get("groups",m.get("roots",[])):
            banned.update([g.get("source_root",g.get("root")),g["split_group"]])
    pilot={r["row_id"] for r in rgz(ART/"20261124_v9_interface/native_model_rollouts.json.gz")}
    for r in rs:
        if r["row_id"] in pilot:banned.update([r["source_group_key"],r["group"]])
    split=rd(OLD/"SPLIT.json")["roots"];pwtrain={r["root"] for r in split if r["partition"]=="train"}
    banned.update(r["root"] for r in split if r["partition"]=="dev")
    # Candidate selection uses only already permitted TRAIN PW groups. Reserve all remaining PW groups for B3.
    aliases={r["source_group_key"]:r["group"] for r in rs if r["tier"]=="core_reference"}
    banned.update(aliases.get(k,k) for k in list(banned))
    return banned,pwtrain,aliases
def select(rs):
    banned,pwtrain,aliases=bound_exposure(rs)
    for r in rs:r["group"]=aliases.get(r["source_group_key"],r["group"])
    chosen={};partition={};domain_groups=defaultdict(set)
    for r in rs:
        if r["group"] in banned or r["source_group_key"] in banned:continue
        if r["domain"]=="PROOFWRITER" and r["group"] not in pwtrain:continue
        if r.get("eval_only"):continue
        domain_groups[r["domain"]].add(r["group"])
    # Assign physical source groups globally before per-domain quotas, including cross-domain business aliases.
    dev=set()
    for dom,gs in sorted(domain_groups.items()):
        if dom=="PROOFWRITER":continue
        groups=sorted(gs,key=lambda g:key(["multi-source-development",g]))
        dev.update(groups[:min(8,len(groups)//4)])
    for dom,gs in sorted(domain_groups.items()):
        for part,gs0 in (("train",gs-dev),("dev",gs&dev)):
            pool=[r for r in rs if r["domain"]==dom and r["group"] in gs0 and not r.get("eval_only") and r["source_group_key"] not in banned]
            seq=list(rr(pool,"mechanism"));grp={}
            for r in seq:
                grp.setdefault(r["group"],[]).append(r)
            ordered=list(grp)[:24 if part=="train" else 8]
            for g in ordered:
                assert partition.setdefault(g,part)==part
                selected=[];used=set()
                for r in rr(grp[g],"tier"):
                    if r["row_id"] in used:continue
                    selected.append(r);used.add(r["row_id"])
                    if len(selected)==4:break
                chosen[dom,g]=selected
    # Include the hold controls only for independently assigned development groups.
    for (dom,g),selected in chosen.items():
        if partition[g]=="dev":
            controls=[r for r in rs if r["domain"]==dom and r["group"]==g and r.get("eval_only")]
            selected.extend(list(rr(controls,"mechanism"))[:2])
    return [(partition[g],r) for (_,g),ss in chosen.items() for r in ss],banned
def pack(r):
    c=r["actor_context"];su=r["supervision"];k=su["kind"]
    menu=r.get("executable_actions",[]);pos=[];neg=[];target=None
    if k=="semantic_pair_dpo":
        pos=[su["chosen_action"]];neg=[su["rejected_action"]];kind="pair"
    elif k=="masked_action_set":
        pos=su["positive_actions"];neg=su["negative_actions"];kind="masked"
    elif k=="multilabel_evidence_selection":
        menu=[{"select_entry":e["entry_id"]} for e in c["evidence_available_from_tools"]]
        pos=[{"select_entry":i} for i in su["positive_entry_ids"]];neg=[{"select_entry":i} for i in su["negative_entry_ids"]];kind="multilabel"
    elif k=="hold_control":pos=su["known_legal_actions"];kind="hold"
    elif r["tier"]=="TASK_ONLY":target=cn(su["target"]);kind="completion"
    else:raise ValueError(k)
    acts=sorted(set(map(cn,menu)));pp=sorted(map(cn,pos));nn=sorted(map(cn,neg))
    assert not (set(pp)&set(nn)) and (set(pp)|set(nn))<=set(acts)
    tag=key([c,acts])
    return tag,dict(ctx=c,actions=acts),dict(kind=kind,pos=pp,neg=nn,target=target)
def main():
    rs,hashes,tot=inputs();sel,banned=select(rs);obs={};labs=[];by=defaultdict(set);seen=set()
    for part,r in sel:
        tag,ob,lab=pack(r)
        if (part,tag,key(lab)) in seen:continue
        seen.add((part,tag,key(lab)));by[part].add(tag)
        obs[tag]=ob;labs.append(dict(tag=tag,root=r["group"],source=r["source_group_key"],row=r["row_id"],domain=r["domain"],tier=r["tier"],mechanism=r["mechanism"],value_tier=r.get("value_tier"),partition=part,**lab))
    assert not (by["train"]&by["dev"])
    gz(OUT/"actor_inputs.gz",obs);gz(OUT/"supervision.gz",labs)
    dump(OUT/"SOURCE_SPLITS.json",dict(input_hashes=hashes,historical_exclusions=sorted(x for x in banned if x),rows=[{k:r[k] for k in ("row","root","source","domain","tier","mechanism","partition","tag")} for r in labs]))
    dump(OUT/"SOURCE_AVAILABLE.json",dict(available=tot,model_input_sha=sha(OUT/"actor_inputs.gz"),labels_sha=sha(OUT/"supervision.gz"),selected=dict(Counter(r["partition"]+"|"+r["domain"] for r in labs)),rule="domain-balanced cap; PW uses existing TRAIN only; hold evaluation only; no fixed-reference target is a rollout"))
    print(len(obs),len(labs),dict(Counter(r["partition"]+"|"+r["domain"] for r in labs)))
if __name__=="__main__":main()

