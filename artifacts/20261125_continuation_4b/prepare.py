"""Bind the extracted cells to native observations and schedule one source-group split."""
import gzip,json,sys,random
from pathlib import Path
from collections import defaultdict,Counter
from hashlib import sha256
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]; ART=ROOT/"artifacts"; TABLE=ART/"20261125_kci_continuation_table_v1"
CORE=Path("D:/research/kci_supervision_alignment_20261123/d_future_core_20261124")
sys.path.insert(0,str(CORE/"tools"))
import execute_proof_core as pw
from importlib.util import spec_from_file_location,module_from_spec
s=spec_from_file_location("typed",CORE.parent/"validation_v9/tools/typed_training.py");typed=module_from_spec(s);s.loader.exec_module(typed)
cn=pw.cn
def js(p):return json.loads(p.read_text(encoding="utf8"))
def gz(p,x):
    with gzip.open(p,"wt",encoding="utf8") as f:json.dump(x,f,ensure_ascii=False)
def readgz(p):
    with gzip.open(p,"rt",encoding="utf8") as f:return json.load(f)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf8")
def strat(r):
    lit=r["actor_input"]["actor_context"]["fixed_norm"]["protected_proposition"]["literal"]
    return (r["stratum"],"attribute" if lit["relation"]=="is" else "binary")
def select(rows):
    old=set(); files={}
    for directory in ("20261124_v9_training","20261124_v9_history_repair","20261125_icml_v02","20261125_icml_v02/B1"):
        path=ART/directory/("freeze.json" if "20261124" in directory else "ROOT_MANIFEST.json")
        m=js(path);files[str(path)]=pw.sh(path)
        for g in m.get("groups",m.get("roots",[])):
            old.add(g["split_group"]);old.add(g.get("source_root",g.get("root")))
    path=ART/"20261124_v9_interface/native_model_rollouts.json.gz"
    ids={r["row_id"] for r in readgz(path)};files[str(path)]=pw.sh(path)
    old.update(r["split_group"] for r in rows if r["row_id"] in ids)
    seen=set(old); bins=defaultdict(list)
    for r in rows:
        if r["split_group"] not in seen:bins[strat(r)].append(r)
    eligible=len({r["split_group"] for b in bins.values() for r in b})
    train=[];dev=[]
    for b in bins.values():b.sort(key=lambda r:pw.h(["continuation-table-7",r["row_id"]]))
    for n,out in ((32,dev),(128,train)):
        while len(out)<n:
            before=len(out)
            for k,b in sorted(bins.items()):
                while b and b[-1]["split_group"] in seen:b.pop()
                if b and len(out)<n:
                    r=b.pop();out.append(r);seen.add(r["split_group"])
            assert len(out)>before
    return train,dev,dict(exclusions=files,eligible=eligible)
def bind_native(rows):
    ids={r["row_id"] for r in rows}
    pre=CORE/"data/PROOFWRITER_CORE_PREREG_V1";mf=js(pre/"MANIFEST.json")
    assert pw.sh(pw.P)==mf["source_sha256"] and pw.sh(pre/"PREREG_MINED_SPECS.jsonl")==mf["spec_sha256"]
    specs={s["spec_id"]:s for s in pw.jl(pre/"PREREG_MINED_SPECS.jsonl") if s["spec_id"] in ids}
    wants={s["original_source_id"] for s in specs.values()};sources={}
    for b in pw.pq.ParquetFile(pw.P).iter_batches(batch_size=512):
        sources.update((x["id"],x) for x in b.to_pylist() if x["id"] in wants)
    return specs,sources
def env_for(r,specs,sources,auth):
    s=specs[r["row_id"]]
    mode=pw.KINDS[int(r["stratum"]=="proofwriter_local_attestation")]
    return pw.Env(sources[s["original_source_id"]],s,mode,licensed=auth)
def context(e,ledger,role):
    c=e.context(ledger)
    if e.licensed:c["fixed_norm"]["rule"]="For this verification task the recipient is explicitly authorized to infer either polarity of the named proposition."
    if role=="A":return c
    visible={k:c[k] for k in ("recipient","task","original_rule_texts","public_rule_logic","logic_semantics","first_slot_eligible_claims","second_slot_required_claim","fixed_norm","recipient_actual_deliveries","communication_contract")}
    visible.update(acting_agent="source_B",role_local_source_records=[dict(literal=pw.literal(e.fs[i]),text=e.src["facts"][i]["text"]) for i in e.holdings["source_B"]])
    return visible
def main():
    assert not (OUT/"FREEZE.json").exists()
    rows=list(pw.jl(TABLE/"CASES.jsonl"));train,dev,manifest=select(rows); rows=train+dev
    cells=defaultdict(dict);ids={r["row_id"] for r in rows}
    for c in pw.jl(TABLE/"EVAL_ONLY_CELLS.jsonl"):
        if c["row_id"] in ids:cells[c["row_id"]][cn(c["first_action"]),cn(c["next_action"])]=c["evaluator_target"]
    specs,sources=bind_native(rows)
    prompts={};graphs=[];schedule=[];audit=Counter(); freq=Counter()
    def reg(c,menu):
        ordered=sorted(menu,key=cn)
        prompt,options=typed.semantic_render(dict(actor_context=c,executable_actions=ordered,supervision={"kind":"semantic_pair_dpo"}))
        tag=pw.h([prompt,options]);prompts[tag]=dict(prompt=prompt,actions=options)
        return tag
    for ix,r in enumerate(rows):
        rid=r["row_id"];meta=specs[rid]
        assert meta["source_group_key"]==r["split_group"]
        g=dict(row_id=rid,root=r["split_group"],stratum=r["stratum"],mechanism=strat(r)[1],partition="train" if ix<len(train) else "dev",worlds={})
        pair=[cn(a) for a in r["annotated_first_actions"]]
        win=r["evaluator_only"]["safe_reference"].index(1);good,bad=pair[win],pair[1-win]
        g["pair"]=[good,bad]
        pool={}
        for norm in ("original","authorized"):
            auth=norm=="authorized";e=env_for(r,specs,sources,auth);rr,_=e.prefix()
            ca=context(e,rr,"A");aa=e.messages("source_A");bb=e.messages("source_B")
            if not auth:assert ca==r["actor_input"]["actor_context"]
            assert ca["authorized_downstream_candidate_catalog"]==bb and aa==r["actor_input"]["executable_actions"]
            head=reg(ca,aa); paths={}
            for a in aa:
                ak=cn(a);ledger,_,va=e.deliver(rr,a);cb=context(e,ledger,"B")
                assert cb["recipient_actual_deliveries"][-1]==a and "authorized_downstream_candidate_catalog" not in cb
                stop={"kind":"SAFE_ABORT","actor":"source_B"}; tag=reg(cb,bb+[stop]);outcomes={}
                for b in bb:
                    bk=cn(b);stored=cells[rid].get((ak,bk))
                    if stored is not None:
                        prefix="authorized_" if auth else ""
                        t=stored[prefix+"task_supported"];v=stored[prefix+"ci_violation"]
                        audit["reused_cells"]+=1
                    if stored is None or ix<3:
                        final,_,vb=e.deliver(ledger,b)
                        actual=int(e.final(final) is not None),int(va["verdict"]!="VALID" or vb["verdict"]!="VALID")
                        if stored is None:t,v=actual;audit["completed_unannotated_cells"]+=1
                        else:assert actual==(t,v);audit["spot_replayed_cells"]+=1
                    outcomes[bk]=2*t+v
                    if stored is not None:pool[norm,ak,bk]=2*t+v
                outcomes[cn(stop)]=int(va["verdict"]!="VALID")
                accepted=[b for b,y in outcomes.items() if y==2] or [cn(stop)]
                paths[ak]=dict(tag=tag,classes=outcomes,accepted=sorted(accepted),reachable=int(2 in outcomes.values()))
            valid=[a for a,p in paths.items() if p["reachable"]]
            assert valid and (bad in valid)==auth and good in valid
            g["worlds"][norm]=dict(head=head,paths=paths,accepted=sorted(valid))
        # Cell IDs are stable and chosen without accessing model choices or scores.
        picked=[];used=set()
        for j in range(4):
            norm="authorized" if j==3 else "original"
            options=[k for k in pool if k[0]==norm and k not in used]
            if not options:
                options=[k for k in pool if k[0]==norm]
                audit["scheduled_repeat_after_catalog_exhausted"]+=1
            ys=sorted({pool[k] for k in options},key=lambda y:(sum(pool[k]==y for k in used if k[0]==norm),freq[norm,y],pw.h([rid,j,y])))
            y=ys[0];k=min((k for k in options if pool[k]==y),key=lambda k:pw.h([rid,j,k]))
            used.add(k);freq[norm,y]+=1
            picked.append(dict(norm=norm,a=k[1],b=k[2],cls=y))
        g["cells"]=picked
        if ix<len(train):
            # Common action slots cover both histories at B and an A decision. Rotate A norm.
            base_norm="authorized" if ix%4==0 else "original"
            common=[(base_norm,None),("original",good),("original",bad),("authorized",bad)]
            # Four stratified auxiliary slots. Oracle uses additional native A/B action practice.
            slots=[]
            for j,(n,a) in enumerate(common):
                w=g["worlds"][n];cell=picked[j];nw=g["worlds"][cell["norm"]]
                alt=cell["a"] if j%2==0 else None
                def task(world,first):
                    item=world if first is None else world["paths"][first]
                    return dict(tag=item.get("head",item.get("tag")),accepted=item["accepted"])
                slots.append(dict(common=task(w,a),oracle=task(nw,alt),cell=cell,head=nw["head"],view=(ix+j)%2))
            schedule.append(dict(root=g["root"],slots=slots))
        graphs.append(g)
    # Reject cross-partition equality of exact native actor inputs or normalized context+menu.
    def tags(gs):
        return {t for g in gs for w in g["worlds"].values() for t in [w["head"]]+[p["tag"] for p in w["paths"].values()]}
    assert not tags(graphs[:len(train)])&tags(graphs[len(train):])
    assert len({g["root"] for g in graphs})==len(graphs)
    assert not ({specs[r["row_id"]]["original_source_id"] for r in train}&{specs[r["row_id"]]["original_source_id"] for r in dev})
    manifest["roots"]=[{k:g[k] for k in ("root","partition","stratum","mechanism","row_id")} for g in graphs]
    dump(OUT/"SPLIT.json",manifest);gz(OUT/"observations.gz",prompts);gz(OUT/"graphs.gz",graphs)
    dump(OUT/"schedule.json",schedule)
    dump(OUT/"LABEL_AUDIT.json",dict(**audit,unique_observations=len(prompts),selected_cell_counts={part:dict(Counter(str(c["cls"]) for g in graphs if g["partition"]==part for c in g["cells"])) for part in ("train","dev")},source_hashes={name:pw.sh(TABLE/name) for name in ("CASES.jsonl","EVAL_ONLY_CELLS.jsonl","SUMMARY.json")},authorization="Both roles see explicit license in licensed native environment. Canonical role B context contains only public rules, received messages, its own holdings and its menu.",accepted="All feasible safe-success actions. On closed B branches only explicit safe exit satisfies the task instruction."))
    print("roots",len(train),len(dev),"counts",audit,"observations",len(prompts))
if __name__=="__main__":main()

