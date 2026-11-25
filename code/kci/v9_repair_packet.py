"""Freeze fresh V9 source groups and bind the two-history experiment to native transitions."""
import argparse
import json
from pathlib import Path
from .paper_worlds import canon, key
from .v9_execution import Execution, read_gz, write_gz
from .v9_interface import Deliveries, sha


def build(delivery, previous, interface, out):
    d=Deliveries(delivery/"deliverables/KCI_VALIDATED_V9_INTERNAL", delivery/"d_future_core_20261124/D_future_core_multisource_v2")
    rows=d.core_reference("proofwriter_local_attestation",purpose="reference_distillation")
    old={v["id"] for v in json.loads((previous/"freeze.json").read_text())["groups"]}
    old.update(v["row_id"] for v in read_gz(interface/"native_model_rollouts.json.gz"))
    retired=[r for r in rows if r["row_id"] in old]
    groups={r["split_group"] for r in retired}
    roots={r["source_group_key"] for r in retired}
    select=[]
    for r in sorted(rows,key=lambda r:key(["pw-repair-v1",r["row_id"]])):
        if r["split_group"] in groups or r["source_group_key"] in roots:continue
        groups.add(r["split_group"]);roots.add(r["source_group_key"]);select.append(r)
        if len(select)==144:break
    assert len(select)==144
    ex=Execution(d,select)
    gs=[]
    for i,r in enumerate(select):
        g=ex.graph(r)
        env=ex.proofs[r["row_id"]]; rr,_=env.prefix()
        g.update(domain="proof",split="fit" if i<64 else "dev" if i<96 else "reserve",source_root=r["source_group_key"],
                 valid_first={canon(a):env.deliver(rr,a)[2]["verdict"]=="VALID" for a in r["executable_actions"]})
        for branches in g["paths"].values():
            for b in branches:
                for o in b["outcomes"].values():
                    for k in ("events","submitted","knowledge_counts"):o.pop(k,None)
        gs.append(g)
    write_gz(out/"graphs.gz",gs);write_gz(out/"prompts.gz",ex.prompts)
    meta=dict(initial=json.loads((previous/"proof_cce_7/result.json").read_text())["weights"],
              groups=[{k:g[k] for k in ("id","source_root","split_group","split")} for g in gs],
              hashes={k:sha(out/k) for k in ("graphs.gz","prompts.gz","PROTOCOL.md")})
    (out/"freeze.json").write_text(json.dumps(meta,indent=2))
    print(len(gs),len(ex.prompts),len(retired))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("delivery","previous","interface","out"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();build(a.delivery,a.previous,a.interface,a.out)

