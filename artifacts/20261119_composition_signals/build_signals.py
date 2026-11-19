"""Bounded KCI relation/permission/task composition DATA audit. No training or API calls.
Generated labels come only from the native executable ReceiptTask environment.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, random, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"code"))
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import targets, intervene
from kci.relation_language import prompt,seed_for
from kci.dynamic_tasks import canonical
PARENT=ROOT/"artifacts"/"20261117_task_rehearsal"
PALETTES=json.loads((PARENT/"palettes.json").read_text(encoding="utf-8"))
ROLES=("Planner","Source","Worker")

def digest(x): return hashlib.sha256(canonical(x).encode("utf-8")).hexdigest()
def dump_rows(p, data):
    p.write_text("".join(canonical(r)+"\n" for r in data),encoding="utf-8")
def load_rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]

def build(out:Path,repeats:int=2):
    if repeats<2 or repeats>4: raise ValueError("Require 2..4 repeats per factor cell to cover TRAIN and DEV")
    if out.exists() and any(out.iterdir()):raise FileExistsError("Refusing to overwrite existing run")
    out.mkdir(parents=True,exist_ok=True)
    worlds,queries,pairs=[],[],[]
    covers=collections.Counter()
    for kind_i,kind in enumerate(("meeting","query")):
      for depth in (1,3):
       for allow in (False,True):
        for required in (False,True):
         for delivered in (False,True):
          for rep in range(repeats):
            idx=len(worlds)//2
            base_id=f"c{idx:04d}"
            seed=910000000 + idx*137 + kind_i
            base=make_world(seed,kind,depth=depth,noise=2,mode="one",
                            allow=allow,required=required,delivered=delivered)
            swapped=intervene(base,"swap")
            split="dev" if rep%2==1 else "train"
            two=[]
            for side,w in enumerate((base,swapped)):
                w["base_group_id"]=base_id
                w["world_id"]=base["world_id"]+("_base" if side==0 else "_swap")
                worlds.append({"base_group_id":base_id,"split":split,"side":side,"world":w})
                cases=targets(w)
                if len(cases)!=5:raise AssertionError("Expected three-role five-state native contract")
                this=[]
                for si,case in enumerate(cases):
                    obs=case["observation"]
                    if obs["role"] not in ROLES:raise AssertionError("Invalid role")
                    palette=PALETTES["train" if split=="train" else "new"]
                    nonce=seed_for(base_id+f":{rep}",ROLES.index(obs["role"]))
                    each={}
                    for ci,candidate in enumerate(case["candidates"]):
                        question,_=prompt(obs,palette,nonce,question="task",candidate=candidate["action"])
                        question=question.replace("A yes; B no.","Reply ANSWER: followed by 1 if possible, otherwise 0.")
                        value=int(candidate["safe_completion"])
                        ref=candidate["value"]
                        if ref is None or len(ref)!=3:raise AssertionError("Missing finite continuation values")
                        record={
                            "id":f"{base_id}:{side}:{si}:{ci}",
                            "base_group_id":base_id,"split":split,"world_id":w["world_id"],
                            "role":obs["role"],"state_index":si,"candidate_index":ci,
                            "action":candidate["action"],"prompt":question,"label":value,
                            "task_value":ref,"immediate_safe":int(candidate["immediate_safe"]),
                            "factors":{"kind":kind,"depth":depth,"allow":allow,
                                       "required":required,"delivered":delivered,"side":side},
                            "proof":{"audit":candidate["audit"],"witness":candidate["witness"],
                                     "source_hashes":case["source_hashes"],"state_id":case["state_id"]},
                            "status":"DIAGNOSTIC_ONLY_NO_TRAINING_APPROVAL"}
                        queries.append(record)
                        each[candidate["action"]]=record
                        covers[(split,obs["role"],value)]+=1
                    this.append(each)
                two.append(this)
            if len(two)!=2:raise AssertionError("Pair missing")
            for state_idx,(a,b) in enumerate(zip(two[0],two[1])):
                if set(a)!=set(b):raise AssertionError("Paired action sets changed")
                for act in sorted(a):
                    x,y=a[act],b[act]
                    if x["role"]!=y["role"]:raise AssertionError("Role changed")
                    flipped=x["label"]!=y["label"]
                    pairs.append({
                        "base_group_id":base_id,"split":split,"role":x["role"],
                        "action":act,"state_index":state_idx,"left_id":x["id"],
                        "right_id":y["id"],"left_label":x["label"],"right_label":y["label"],
                        "relation_flip":flipped,
                        "factors":{k:v for k,v in x["factors"].items() if k!="side"},
                        "status":"PAIRED_DIAGNOSTIC_ONLY"})
    grouped=collections.defaultdict(list)
    for r in queries:grouped[r["base_group_id"]].append(r)
    seen=set()
    for p in pairs:
        if p["left_id"] in seen or p["right_id"] in seen:raise AssertionError("Duplicate paired row")
        seen.add(p["left_id"]);seen.add(p["right_id"])
    if len(seen)!=len(queries):raise AssertionError("Not all query rows paired")
    if {r["id"] for r in queries}!=seen:raise AssertionError("Incomplete query coverage")
    split_groups=collections.defaultdict(set)
    for p in pairs:split_groups[p["base_group_id"]].add(p["split"])
    if any(len(v)!=1 for v in split_groups.values()):raise AssertionError("Root leaks across train/dev")
    byfact=collections.Counter((p["split"],p["factors"]["allow"],p["factors"]["required"],
                                p["factors"]["delivered"],p["relation_flip"]) for p in pairs)
    byrole=collections.Counter((p["split"],p["role"],p["relation_flip"]) for p in pairs)
    summary={
        "version":"20261119-native-composition-diagnostic-v1",
        "root_count":len(grouped),"world_count":len(worlds),
        "queries":len(queries),"pairs":len(pairs),
        "train_roots":len([v for v in split_groups.values() if "train" in v]),
        "dev_roots":len([v for v in split_groups.values() if "dev" in v]),
        "train_queries":sum(r["split"]=="train" for r in queries),
        "dev_queries":sum(r["split"]=="dev" for r in queries),
        "positive":sum(r["label"]==1 for r in queries),
        "negative":sum(r["label"]==0 for r in queries),
        "flip_pairs":sum(p["relation_flip"] for p in pairs),
        "hold_pairs":sum(not p["relation_flip"] for p in pairs),
        "by_role":{str(k):v for k,v in sorted(byrole.items())},
        "by_factor":{str(k):v for k,v in sorted(byfact.items())},
        "group_disjoint":True,
        "source_task_families":["meeting","query"],
        "native_contract":"receipt_task_v1_synthetic_obligation",
        "independent_business_generalization":False,
        "natural_norm_validity_verified":False,
        "training_ready":False,
        "gpu_training":"NO_GO"}
    for name,data in (("worlds",worlds),("queries",queries),("pairs",pairs)):
        dump_rows(out/(name+".jsonl"),data)
    (out/"quality.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*")) if p.is_file()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return summary

def audit(out:Path):
    manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    for name,hashval in manifest.items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=hashval:raise AssertionError("Tampered "+name)
    q=load_rows(out/"queries.jsonl");p=load_rows(out/"pairs.jsonl")
    lookup={r["id"]:r for r in q}
    for x in p:
        a,b=lookup[x["left_id"]],lookup[x["right_id"]]
        assert a["base_group_id"]==b["base_group_id"]==x["base_group_id"]
        assert a["split"]==b["split"]==x["split"]
        assert a["label"]==x["left_label"] and b["label"]==x["right_label"]
        assert x["relation_flip"]==(a["label"]!=b["label"])
        assert a["action"]==b["action"] and a["role"]==b["role"]
    assert all(x["status"]=="DIAGNOSTIC_ONLY_NO_TRAINING_APPROVAL" for x in q)
    return {"checks":"PASS","queries":len(q),"pairs":len(p),"training_ready":False}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["build","audit"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=2)
    a=parser.parse_args()
    result=build(a.output,a.repeats) if a.command=="build" else audit(a.output)
    print(json.dumps(result,ensure_ascii=False))
