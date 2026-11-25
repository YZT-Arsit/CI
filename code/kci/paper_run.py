"""GPU verification entrypoint: prepare, collect, update and independently reload one shared policy."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean
import sys

from .paper_alignment import Alignment, metrics, prompt
from .paper_policy import SharedLM
from .paper_worlds import PhaseWorld, RelWorld, canon, filehash, key, phase, relational_module, worlds


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf8")


def pack(research,out,cohort="verification"):
    eligible = list(worlds(research))
    by = defaultdict(dict)
    for w in eligible:
        family="phase2" if isinstance(w,PhaseWorld) else w.env.family
        by[family].setdefault(w.root,[]).append(w)
    used=set();records=[]
    root_split={}
    for fam in ("RARE_SET_INTERSECTION","phase2","TWO_HOP_JOIN","DIRECT_JOIN"):
        roots=sorted(by[fam],key=lambda r:key(["paper-validation-01",fam,r]))
        if cohort=="verification":
            roots=[r for r in roots if r not in used][:1 if fam=="RARE_SET_INTERSECTION" else 3]
        for i,root in enumerate(roots):
            if cohort=="full":
                split=root_split.setdefault(root,"validation" if i%5==2 else "fit")
            else:
                split="validation" if i==2 else "fit"
            used.add(root)
            variants=by[fam][root]
            if fam!="phase2" and cohort=="verification":
                first=min(w.env.mechanism_id for w in variants)
                variants=[w for w in variants if w.env.mechanism_id==first]
            for w in variants:
                if fam=="phase2":
                    e=w.env
                    data=dict(authorized=e.authorized,reorder=e.reorder,swap=e.swap,variant=e.variant)
                else:
                    e=w.env;s=e.spec
                    # Only the tables needed to reconstruct the native projections.
                    tables={}
                    for role in ["public","sensitive"]+(["bridge"] if fam=="TWO_HOP_JOIN" else []):
                        owner=s[role+"_owner"];table=s[role+"_table"]
                        tabs=tables.setdefault(owner,{"table":{}})["table"]
                        tabs[table]=e.source["data_distribution"][owner]["table"][table]
                    source=dict(data_distribution=tables,run_1_benign={"alice_goal":e.model["external_benign_goal_context"]})
                    data=dict(spec=s,source=source,authorized=e.preauthorized)
                records.append(dict(root=root,world=w.id,family=fam,split=split,data=data))
    out.mkdir(parents=True,exist_ok=False)
    with gzip.open(out/"packet.json.gz","wt",encoding="utf8") as f:json.dump(records,f,separators=(",",":"))
    native=research/"kci_diverse_relational_20261124/code/relational_env.py"
    (out/"relational_env.py").write_bytes(native.read_bytes())
    dump(out/"manifest.json",dict(packet=filehash(out/"packet.json.gz"),native=filehash(native),
            worlds=[{k:v for k,v in x.items() if k!="data"} for x in records],
            cohort=cohort,source_partition="V8 TRAIN",eligible_worlds=len(eligible),
            eligible_roots=len({w.root for w in eligible}),seed=7,horizon=24))
    return records


def unpack(folder):
    mf=json.loads((folder/"manifest.json").read_text(encoding="utf8"))
    if filehash(folder/"packet.json.gz")!=mf["packet"] or filehash(folder/"relational_env.py")!=mf["native"]:
        raise ValueError("Packet integrity failure")
    with gzip.open(folder/"packet.json.gz","rt",encoding="utf8") as f:records=json.load(f)
    if [{k:v for k,v in r.items() if k!="data"} for r in records]!=mf["worlds"]:raise ValueError("Split mismatch")
    ids=defaultdict(set);specs={s.id:s for s in phase.make_workflows()}
    for row in records:
        ids[row["split"]].add(row["root"])
    if ids["fit"] & ids["validation"]:raise ValueError("Source group overlap")
    for row in records:
        d=row["data"]
        if row["family"]=="phase2":w=PhaseWorld(specs[row["root"]],**d)
        else:w=RelWorld(relational_module(folder/"relational_env.py"),**d)
        if row["world"]!=w.id or row["root"]!=w.root:raise ValueError("Source replay mismatch")
        yield row["split"],w


def sample_world(engine,seed):
    traces=[engine.visit(seed)]
    # Sample FIT supervision also at lawful history prefixes reached by the frozen
    # reference. A forced prefix is never passed off as an actual model failure.
    traces.append(engine.visit(seed,"reference"))
    if isinstance(engine.world,RelWorld):
        def early(ob):return ob["role_stage"]=="public"
        def alternate(ob):return next(a for a in ob["legal_actions"] if a["kind"]==engine.world.mod.PRIVATE)
    else:
        def early(ob):return ob["stage"]=="A_SEND"
        def alternate(ob):return "A_SEND_PACKAGE_B"
    traces.append(engine.visit(seed,"reference_prefix",first=(early,alternate)))
    return traces


def score(worlds_list,model):
    results=[]
    for w in worlds_list:
        decisions={}
        def choose(ob):
            action=model.choose(ob)
            decisions[key(prompt(ob))]=action
            return action
        engine=Alignment(w,choose,model.version)
        v,_=engine.outcome(w.initial(),engine.horizon,"current")
        results.append(dict(world=w.id,root=w.root,decision_sha256=key(decisions),metrics=metrics(v)))
    per_root=defaultdict(list)
    for r in results:per_root[r["root"]].append(r["metrics"])
    def macro(k):return mean(mean(r[k] for r in vs) for vs in per_root.values())
    totals={k:macro(k) for k in results[0]["metrics"] if k!="cost_given_safe_success"}
    totals["cost_given_safe_success"]=totals["safe_success_cost_mass"]/totals["stsr"] if totals["stsr"] else None
    totals["ci_compliance_lower"]=max(0.,1-totals["any_ci"]-totals["unknown"])
    totals["ci_compliance_upper"]=1-totals["any_ci"]
    return dict(policy=model.version,roots=len(per_root),worlds=results,macro=totals)


def run(args):
    from .paper_worlds import filehash
    import torch
    import gc
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    pool=list(unpack(args.packet)); train=[w for sp,w in pool if sp=="fit"];val=[w for sp,w in pool if sp=="validation"]
    model=SharedLM(args.base,args.adapter)
    dump(out/"config.json",dict(packet=filehash(args.packet/"manifest.json"),initial=model.version,steps=args.steps,
          run="single_shared_adapter_one_iteration",device=str(model.device),torch=torch.__version__,
          gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
          code={p.name:filehash(p) for p in Path(__file__).parent.glob("*.py") if
                p.name.startswith("paper_") or p.name in ("v4_phase2_workflows.py","diverse_ci_branch_env.py")}))
    allrows=[]
    with gzip.open(out/"fit_evidence.jsonl.gz","wt",encoding="utf8") as ev, gzip.open(out/"fit_traces.jsonl.gz","wt",encoding="utf8") as tr:
        for i,w in enumerate(train):
            engine=Alignment(w,model.choose,model.version)
            traces=sample_world(engine,7)
            for t in traces:tr.write(canon(t)+"\n")
            for r in engine.evidence():ev.write(canon(r)+"\n")
            tr.write(canon(dict(world=w.id,failures=engine.failures))+"\n")
            allrows.extend(engine.pairs.values())
            print("mined",i+1,len(train),w.root,dict(Counter(r["source"] for r in engine.pairs.values())),flush=True)
    with gzip.open(out/"preferences.jsonl.gz","wt",encoding="utf8") as f:
        for r in allrows:f.write(canon(r)+"\n")
    with gzip.open(out/"fit_policy.jsonl.gz","wt",encoding="utf8") as f:
        for observation,action in model.cache.items():
            f.write(canon(dict(observation_id=key(observation),action=action))+"\n")
    before=score(val,model);dump(out/"before.json",before)
    training=model.train(allrows,out/"shared_adapter",steps=args.steps)
    dump(out/"optimizer.json",training)
    after=score(val,model);dump(out/"after.json",after)
    del engine,model
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    restored=SharedLM(args.base,out/"shared_adapter")
    checked=score(val,restored)
    if checked!=after:raise RuntimeError("Reload changed end-to-end decisions or weights")
    dump(out/"complete.json",dict(rows=dict(Counter(x["source"] for x in allrows)),fit_roots=len({w.root for w in train}),
         validation_roots=after["roots"],reload_equal=True,before=before["macro"],after=after["macro"],
         weights=restored.version))
    print("VERIFIED",flush=True)


def ablate(args):
    """Replay the frozen preference pool with the same starting weights and update budget."""
    def read(name):
        return json.loads((args.collection/name).read_text(encoding="utf8"))
    config=read("config.json")
    if config["packet"]!=filehash(args.packet/"manifest.json"):
        raise ValueError("Wrong collection packet")
    pool=list(unpack(args.packet))
    val=[w for split,w in pool if split=="validation"]
    with gzip.open(args.collection/"preferences.jsonl.gz","rt",encoding="utf8") as f:
        rows=[json.loads(line) for line in f]
    if any(row["world"] not in {w.id for split,w in pool if split=="fit"} for row in rows):
        raise ValueError("Preferences contain non-fit worlds")
    arms={"ci":("ci",),"ci_future":("ci","future"),"ci_repair":("ci","repair")}
    selected=arms[args.arm]
    if not set(selected)<={r["source"] for r in rows}:raise ValueError("Missing source for ablation")
    model=SharedLM(args.base,args.adapter)
    if model.version!=config["initial"]:raise ValueError("Wrong initial weights")
    args.output.mkdir(parents=True,exist_ok=False)
    weights={k:float(k in selected) for k in ("ci","future","repair")}
    training=model.train(rows,args.output/"shared_adapter",steps=config["steps"],lambdas=weights)
    dump(args.output/"optimizer.json",training)
    dump(args.output/"after.json",score(val,model))


def main():
    ap=argparse.ArgumentParser(__doc__);cmd=ap.add_subparsers(dest="command",required=True)
    prep=cmd.add_parser("prepare");prep.add_argument("--research",type=Path,required=True);prep.add_argument("--output",type=Path,required=True)
    job=cmd.add_parser("verify");job.add_argument("--packet",type=Path,required=True);job.add_argument("--base",type=Path,required=True)
    job.add_argument("--adapter",type=Path,required=True);job.add_argument("--output",type=Path,required=True)
    job.add_argument("--steps",type=int,default=16)
    ab=cmd.add_parser("ablate")
    for name in ("packet","collection","base","adapter","output"):
        ab.add_argument("--"+name,type=Path,required=True)
    ab.add_argument("--arm",choices=("ci","ci_future","ci_repair"),required=True)
    prep.add_argument("--cohort",choices=("verification","full"),default="verification")
    a=ap.parse_args()
    if a.command=="prepare":pack(a.research,a.output,a.cohort)
    elif a.command=="verify":run(a)
    else:ablate(a)
if __name__=="__main__":main()

