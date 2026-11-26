"""Build the internal KCI role-action release from pinned local deliveries."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from io import TextIOWrapper
from pathlib import Path
import gzip
import json
import sys

import sources as s
from build_cache import build, CACHE, NAMES
from native_rows import cn, lines, legacy as p

sys.path.insert(0,str(p.BASE/"validation_v9/tools"))
from build_validated_v9 import actor_context

OUT = s.ART/"20261125_unified_roles_v1"


def sha_pins():
    verified={}
    def pin(file, expected=None):
        h=p.sha(file)
        if expected and h!=expected:
            raise ValueError(f"Source hash mismatch {file}")
        verified[str(file)]=h
    for base,names in [(s.V9,["typed/"+f.name for f in (s.V9/"typed").glob("*.jsonl")]+["validation/SOURCE_SPLITS.json","MANIFEST.json"]),
                       (s.CORE/"D_future_core_multisource_v2",["D_future_core.jsonl","MANIFEST.json"]),
                       (s.TASK,["task_support/TRAIN_SOURCE_GROUNDED_TASK_ONLY.jsonl"]),
                       (s.n.data/"EXISTING_602_CORE_AUDIT_V1",["EVAL_ONLY_REPLAYED_CORE_AND_ABLATION.jsonl.gz"]),
                       (s.n.data/"AGGREGATE_K_CORE_V1",["PREREG_ALL_CANDIDATES.jsonl"])]:
        inventory=p.rd(base/"FILE_SHA256.json")
        for name in names:pin(base/name,inventory[name])
    ng=s.n.data/"NEGOTIATION_CORE_V2"
    pin(ng/"EVAL_ONLY_ALL_NATIVE_BRANCHES.jsonl.gz",p.rd(ng/"MANIFEST.json")["witness_sha256"])
    pin(s.DG/"DPO_R_TASK_AND_CI_TRAIN.jsonl",p.rd(s.DG/"SUMMARY.json")["file_sha256"])
    pin(s.PHASE/"SHARED_ROLE_ACTION_SFT_TRAIN_ONLY.jsonl",p.rd(s.PHASE/"SFT_WITNESS_MANIFEST.json")["SFT_rows_sha256"])
    pin(s.API/"REVIEWED_TRAIN_CANDIDATES.jsonl",p.rd(s.API/"REVIEWED_EXPORT_SUMMARY.json")["output_sha256"])
    for directory,name,inventory in s.V4_FILES:
        pin(s.V4/directory/name,p.rd(s.V4/directory/inventory)[name])
    for f in [s.SELECT/"TRAIN_DEMOS.gz",s.SELECT/"EVIDENCE_AUDIT.json",s.n.data/"NEGOTIATION_CORE_INDEPENDENT_V3/SOURCE_ALIASES.jsonl",
              s.n.dynamic.aud.REL/"data/PREREG_RELATIONAL_TASKS_V2/PREREG_TRAIN_MECHANISMS_WITH_QUERIES.jsonl",
              Path(__file__),Path(s.__file__),p.BASE/"validation_v9/tools/build_validated_v9.py"]:
        pin(f)
    return verified


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    pins=sha_pins()
    rs,_,availability=p.inputs()
    core={r["row_id"]:r for r in rs if r["tier"]=="core_reference"}
    groups,flags,registry,split_pins=s.split_registry(rs)
    # New validation is held out from this release. Historical development and
    # previously used fitting roots cannot become independent model confirmation.
    exposed={groups.find(k) for k in p.rd(p.NEW/"fresh_freeze.json")["excluded_groups"]}
    candidates=defaultdict(set)
    for r in rs:
        g=groups.find(r["group"])
        if not flags[g] and g not in exposed and r["tier"]!="controls_only":
            candidates[r["source_dataset"]].add(g)
    valid=set()
    for domain, roots in candidates.items():
        roots=sorted(roots,key=lambda g:p.key(["unified_roles_holdout_v1",domain,g]))
        if len(roots)>=2:
            valid.update(roots[:max(1,len(roots)//12)])
    cache={}
    for name in NAMES:
        cache[name]=build(name)
    seen=defaultdict(dict)
    lineage=[]
    contexts={}
    omitted=Counter()
    counts=Counter()
    by_tier=Counter()
    def split(g):
        states=flags[g]
        if states & {"locked_test","prospective"}:return "sealed"
        if "dev" in states:
            return "development_exposed" if "train_used" in states else "development"
        return "validation" if g in valid else "train"
    def data():
        yield from s.typed(rs)
        yield from s.supplementary(core)
        for name in NAMES:
            yield from lines(CACHE/(name+".jsonl.gz"))
    for r in data():
        g=groups.find(r["root"])
        part=split(g); route=r["route"];tier=r["tier"]
        by_tier[tier,part]+=1
        if part=="sealed":
            omitted["reserved_source"]+=1
            continue
        ctx=r["ob"]["ctx"];menu=r["ob"]["actions"]
        if tier in ("relational_reference_witness","pisas_reference_witness","scope_downstream_witness","temporal_downstream_witness"):
            ctx=actor_context(ctx)
        if tier=="full_team_task_witness":
            ctx=actor_context(ctx)
        assert not set(ctx)&{"evaluator_target","terminal_outcome","true_agent_trace","protected_proof_eval_only","Qstar","Qpi"}
        actor=ctx.get("acting_agent") or ctx.get("actor") or ctx.get("role") or ctx.get("sender_name")
        assert actor
        role=ctx.get("role_stage") or ctx.get("role") or ("enterprise_sender" if tier=="train_static_evidence_selection" else actor)
        obj=dict(actor_context=ctx,executable_actions=[json.loads(a) for a in menu])
        tag=p.key(obj);contexts.setdefault(tag,obj)
        pp,nn=r["positive"],r["negative"]
        if menu and route!="process_sft":
            assert (set(pp)|set(nn))<=set(menu)
        if route=="process_sft":
            assert all(cn(json.loads(a)["action"]) in menu for a in pp)
        assert pp and not set(pp)&set(nn)
        if route.endswith("dpo"): assert len(pp)==len(nn)==1
        # Same state can have multiple valid teacher witnesses. Pool positive-only
        # demonstrations at the same role state, preserve every supporting path.
        witness=r["evidence"]=="executed_safe_success_witness"
        sig=p.key([g,tag,route,None if witness else pp,nn])
        record=seen[part].setdefault(sig,dict(record_id=sig,group=g,source=r["source"],domain=r["domain"],actor=actor,
                                               role=role,route=route,tag=tag,accepted=[],rejected=nn,tiers=[],evidence=[],value_tiers=[]))
        if tier not in record["tiers"]:record["tiers"].append(tier)
        if r["evidence"] not in record["evidence"]:record["evidence"].append(r["evidence"])
        dimension=r["detail"].get("value_tier")
        if dimension and dimension not in record["value_tiers"]:record["value_tiers"].append(dimension)
        record["accepted"]=sorted(set(record["accepted"])|set(pp))
        lineage.append(dict(record=sig,source=r["source"],row=r["row"],tier=tier,detail=r["detail"],partition=part))
    # Source components are already disjoint. Enforce exact observation isolation
    # too, including across loss types. Drop ambiguous cross-partition prompts.
    tags=defaultdict(set)
    for part,recs in seen.items():
        for r in recs.values():tags[r["tag"]].add(part)
    collisions={t for t,parts in tags.items() if len(parts)>1}
    def gzfile(name,stack):
        raw=stack.enter_context((out/(name+".jsonl.gz")).open("wb"))
        zipped=stack.enter_context(gzip.GzipFile(filename="",fileobj=raw,mode="wb",compresslevel=5,mtime=0))
        return stack.enter_context(TextIOWrapper(zipped,encoding="utf8",newline="\n"))
    files={}; groups_out=defaultdict(set); contexts_out=defaultdict(set); by_role=Counter()
    emitted=set(); contradictions=set(); edges=defaultdict(set); closed=defaultdict(set)
    for part,recs in seen.items():
        for r in recs.values():
            t=r["tag"]
            if r["route"].endswith("dpo"):
                edges[(part,t,r["route"])].add((r["accepted"][0],r["rejected"][0]))
            if "closed_action_set" in r["evidence"]:
                closed[part,t].add(tuple(r["accepted"]))
    for (part,t,_),pairs in edges.items():
        if any((b,a) in pairs for a,b in pairs):contradictions.add((part,t))
    for k,v in closed.items():
        if len(v)>1:contradictions.add(k)
    with ExitStack() as stack:
        for part,recs in sorted(seen.items()):
            for sig,r in sorted(recs.items()):
                t=r["tag"];route=r["route"]
                if t in collisions or (part,t) in contradictions:
                    omitted["cross_partition_observation" if t in collisions else "incompatible_labels"]+=1
                    continue
                bucket="controls" if route=="controls" else part
                name=bucket+"_"+route if bucket!="controls" else "controls"
                if name not in files:files[name]=gzfile(name,stack)
                emitted.add(sig)
                item=dict(**r,partition=part,**contexts[t])
                files[name].write(cn(item)+"\n")
                groups_out[part].add(r["group"]);contexts_out[part].add(t)
                counts[name]+=1
                by_role[part,route,r["domain"],str(r["role"])]+=1
        trace=gzfile("provenance",stack)
        for l in lineage:
            l["exported"]=l["record"] in emitted
            trace.write(cn(l)+"\n")
    for part,roots in groups_out.items():
        for other in groups_out:
            if other!=part:
                assert roots.isdisjoint(groups_out[other]) and contexts_out[part].isdisjoint(contexts_out[other])
    p.dump(out/"SPLITS.json",dict(components={g:sorted(k for k in aliases) for g,aliases in invert(groups).items()},
            partitions={part:sorted(gs) for part,gs in groups_out.items()},historical_and_sealed_registry=registry,
            prospective_metadata=p.rows(s.V9/"validation/PROSPECTIVE_24_ORIGINAL_SOURCE_METADATA_ONLY.jsonl"),
            new_validation="Source-group holdout for this release only. Prior uncontrolled exposure is unknown. No sealed outcomes loaded."))
    p.dump(out/"COVERAGE.json",dict(typed_source_supply=availability,raw_by_tier_partition={"|".join(k):v for k,v in sorted(by_tier.items())},
            by_role_route_domain={"|".join(k):v for k,v in sorted(by_role.items())},omitted=dict(omitted),
            conflicting_observation_hashes=sorted(contradictions),cross_partition_observation_hashes=sorted(collisions)))
    p.dump(out/"MANIFEST.json",dict(format="kci-role-actions-v1",release="internal research",shards={
        name:dict(rows=qty,sha256=p.sha(out/(name+".jsonl.gz"))) for name,qty in sorted(counts.items())},
        group_counts={k:len(v) for k,v in groups_out.items()},source_hashes={**pins,**split_pins},native_bindings=cache,
        metadata_sha256={name:p.sha(out/name) for name in ["COVERAGE.json","SPLITS.json","provenance.jsonl.gz"]},
        training_prefix="train_",controls_gradient=False,new_target_policy_values=0))
    print(dict(sorted(counts.items())),dict(omitted),flush=True)


def invert(groups):
    result=defaultdict(list)
    for x in list(groups.parent):result[groups.find(x)].append(x)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=OUT)
    main(parser.parse_args().out)

