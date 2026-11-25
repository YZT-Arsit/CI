"""Freeze the V9 source-group panel and replay all native leaves before GPU fitting."""
import argparse
from collections import Counter
from pathlib import Path
import json

from .v9_execution import Execution,read_gz,write_gz
from .v9_interface import Deliveries,sha
from .paper_worlds import canon,key


def prepare(root,out,pilot):
    d=Deliveries(root/"deliverables/KCI_VALIDATED_V9_INTERNAL",root/"d_future_core_20261124/D_future_core_multisource_v2")
    retired={r["row_id"] for r in read_gz(pilot/"native_model_rollouts.json.gz")}
    selected=[];split={}
    for source,n,m in (("proofwriter_local_attestation",16,16),("negotiation",12,8)):
        rows=d.core_reference(source,purpose="reference_distillation")
        old=[r for r in rows if r["row_id"] in retired]
        used={r["split_group"] for r in old};roots={r["source_group_key"] for r in old}
        keep=[]
        for r in sorted(rows,key=lambda x:key(["v9-algo-01",x["row_id"]])):
            if r["split_group"] in used or r["source_group_key"] in roots:continue
            used.add(r["split_group"]);roots.add(r["source_group_key"]);keep.append(r)
            if len(keep)==n+m:break
        assert len(keep)==n+m
        for i,r in enumerate(keep):split[r["row_id"]]="fit" if i<n else "dev"
        if source=="negotiation":
            for r in old:split[r["row_id"]]="interface"
            keep+=old
        selected+=keep
    ex=Execution(d,selected)
    graphs=[];bindings=[]
    for r in selected:
        g=ex.graph(r)
        proof=r["stratum"].startswith("proof")
        g.update(split=split[r["row_id"]],source_root=r["source_group_key"],domain="proof" if proof else "neg")
        bindings.append({k:g[k] for k in ("id","source_root","split_group","domain","split")})
        if proof:
            e=ex.proofs[r["row_id"]]
            rr,_=e.prefix()
            g["valid_first"]={canon(a):e.deliver(rr,a)[2]["verdict"]=="VALID" for a in r["executable_actions"]}
        else:g["valid_first"]={a:True for a in g["paths"]}
        for branches in g["paths"].values():
            for b in branches:
                for v in b["outcomes"].values():
                    for k in ("events","submitted","knowledge_counts"):v.pop(k,None)
        graphs.append(g)
    by={s:{r["source_root"] for r in bindings if r["split"]==s} for s in ("fit","dev","interface")}
    assert not (by["fit"]&by["dev"] or by["interface"]&(by["dev"]|by["fit"]))
    write_gz(out/"graphs.gz",graphs);write_gz(out/"prompts.gz",ex.prompts)
    meta=dict(groups=bindings,initial="eb5fa1e2e0cb5e6f282ed30d8a23388299f397737df1fa6e0f9079c868e27390",
              base_shards=json.loads((pilot/"model_choices.json").read_text(encoding="utf8"))["base_shards"],
              hashes={f:sha(out/f) for f in ("graphs.gz","prompts.gz","PROTOCOL.md")})
    (out/"freeze.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    print(dict(Counter((r["domain"],r["split"]) for r in graphs)),len(ex.prompts))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    for k in ("root","out","pilot"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();prepare(a.root,a.out,a.pilot)

