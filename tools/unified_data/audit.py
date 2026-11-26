"""Independently read the release through its training interface and build the evaluation index."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from train import examples,prompt,execution_role,ROUTES


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf8")


def audit(path):
    manifest=json.loads((path/"MANIFEST.json").read_text(encoding="utf8"))
    split=json.loads((path/"SPLITS.json").read_text(encoding="utf8"))
    reverse={alias:group for group,members in split["components"].items() for alias in members}
    contexts=defaultdict(set);prompts=defaultdict(set);roots=defaultdict(set)
    tiers=Counter();domains=defaultdict(set);bound=Counter();held=defaultdict(lambda:defaultdict(set))
    ids={};roles=Counter();rowcounts=Counter();vals=Counter()
    for name,spec in manifest["shards"].items():
        file=path/(name+".jsonl.gz")
        assert hashlib.sha256(file.read_bytes()).hexdigest()==spec["sha256"]
        with gzip.open(file,"rt",encoding="utf8") as fh:
            for line in fh:
                r=json.loads(line);part=r["partition"];route=r["route"]
                assert name==("controls" if route=="controls" else part+"_"+route)
                assert r["group"] in split["partitions"][part]
                assert reverse[r["source"]]==r["group"]
                sig=r["record_id"];assert sig not in ids;ids[sig]=part
                visible={k:r[k] for k in ("actor_context","executable_actions")}
                assert digest(visible)==r["tag"]
                model_prompt=prompt(r)
                assert model_prompt==prompt(dict(visible,route=route))
                roots[part].add(r["group"]);contexts[digest(r["actor_context"])].add(part);prompts[r["tag"]].add(part)
                menu={digest(a) for a in r["executable_actions"]}
                pp=set(r["accepted"]);nn=set(r["rejected"])
                assert pp and not pp&nn
                if route.endswith("dpo"):assert len(pp)==len(nn)==1
                for label in pp|nn:
                    a=json.loads(label)
                    if route=="process_sft":a=a["action"]
                    if menu:assert digest(a) in menu
                if route=="static_evidence":
                    values={e["entry_id"] for e in r["actor_context"]["evidence_available_from_tools"]}
                    assert set(json.loads(next(iter(pp)))["selected_entry_ids"])<=values
                rowcounts[name]+=1
                if route=="controls":continue
                upper=len(model_prompt.encode())+max(len(s.encode()) for s in pp|nn)
                bound[route]=max(bound[route],upper)
                domains[part,r["domain"]].add(r["group"])
                roles[part,r["domain"],execution_role(r),route]+=1
                for t in r["tiers"]:tiers[part,t]+=1
                for v in r["value_tiers"]:vals[part,v]+=1
                if part=="validation":
                    g=held[r["group"]];g["domains"].add(r["domain"]);g["shards"].add(name);g["tiers"].update(r["tiers"])
                    g["source_roots"].add(r["source"])
            assert rowcounts[name]==spec["rows"]
    blocked={reverse.get(rec["source"],rec["source"]) for rec in split["historical_and_sealed_registry"] if rec["status"] in ("prospective","locked_test")}
    assert all(not blocked&gs for gs in roots.values())
    assert all(len(v)==1 for v in contexts.values()) and all(len(v)==1 for v in prompts.values())
    for part,gs in roots.items():
        assert all(not gs&v for other,v in roots.items() if other!=part)
    for r in ROUTES:
        assert sum(1 for _ in examples(path,r))==rowcounts["train_"+r]
    for name in ("validation","controls"):
        try:next(examples(path,name))
        except ValueError:pass
        else:raise AssertionError("Non-training label reached the trainer")
    for name,h in manifest["metadata_sha256"].items():
        assert hashlib.sha256((path/name).read_bytes()).hexdigest()==h
    refs=set()
    with gzip.open(path/"provenance.jsonl.gz","rt",encoding="utf8") as f:
        for line in f:
            r=json.loads(line)
            if not r["exported"]:continue
            assert ids[r["record"]]==r["partition"];refs.add(r["record"])
            if r["partition"]=="validation":
                g=reverse[r["source"]];held[g]["source_rows"].add(r["row"])
    assert refs==ids.keys()
    save(path/"VALIDATION_INDEX.json",{g:{k:sorted(v) for k,v in fields.items()} for g,fields in sorted(held.items())})
    out=dict(rows=dict(rowcounts),train_records=sum(v for k,v in rowcounts.items() if k.startswith("train_")),
             validation_records=sum(v for k,v in rowcounts.items() if k.startswith("validation_")),
             source_groups={k:len(v) for k,v in roots.items()},groups_per_domain={"|".join(k):len(v) for k,v in domains.items()},
             tiers={"|".join(k):v for k,v in tiers.items()},core_value_dimensions={"|".join(k):v for k,v in vals.items()},
             max_prompt_plus_target_utf8_bytes=dict(bound),traceable_records=len(refs),unique_actor_contexts=len(contexts),
             cross_partition_contexts=0,cross_partition_menus=0,cross_partition_groups=0,sealed_metadata_groups_in_shards=0,
             historical_and_prospective_status_counts=dict(Counter(r["status"] for r in split["historical_and_sealed_registry"] if r["status"] in ("locked_test","prospective"))),
             manifest_sha256=hashlib.sha256((path/"MANIFEST.json").read_bytes()).hexdigest())
    save(path/"EXECUTION_ROLE_COUNTS.json",{"|".join(k):v for k,v in sorted(roles.items())})
    save(path/"AUDIT.json",out)
    print(out["train_records"],out["source_groups"],out["max_prompt_plus_target_utf8_bytes"])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("data",type=Path)
    audit(p.parse_args().data)

