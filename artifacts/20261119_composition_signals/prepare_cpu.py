"""Freeze matched CPU schedules before any model call. No GPU, no API.
This is a *batching/curriculum* control, not a new algorithm.
"""
from __future__ import annotations
import hashlib,json,random,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COMP=ROOT/"artifacts"/"20261117_composition_repair"
sys.path.insert(0,str(ROOT/"code"))
import build_signals
from kci.dynamic_tasks import canonical

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump_jsonl(path,data):
    if path.exists():raise FileExistsError(path)
    path.write_text("".join(canonical(x)+"\n" for x in data),encoding="utf-8")

def select_dev(out, train):
    qs={x["id"]:x for x in build_signals.load_rows(out/"queries.jsonl")}
    pairs=build_signals.load_rows(out/"pairs.jsonl")
    selected = []
    for p in pairs:
        if p["split"]=="dev" and p["relation_flip"]:
            selected.append(p)
    holds = [p for p in pairs if p["split"]=="dev" and not p["relation_flip"]]
    used=set(); label_count=Counter()
    for p in list(selected):
        cs=[h for h in holds if h["role"]==p["role"] and h["base_group_id"] not in used]
        if not cs:raise AssertionError("Not enough DEV controls")
        def key(h):
            x,y=qs[h["left_id"]],qs[h["right_id"]]
            assert x["label"]==y["label"]
            return (label_count[x["label"]],h["base_group_id"]==p["base_group_id"],h["base_group_id"],h["state_index"])
        h=min(cs,key=key);used.add(h["base_group_id"])
        label_count[qs[h["left_id"]]["label"]]+=1;selected.append(h)
    rows=[]
    for pair in selected:
        for side in ("left_id","right_id"):
            x=qs[pair[side]]
            rows.append({**x,"pair_group":pair["base_group_id"]+":"+str(pair["state_index"])+":"+pair["action"],
                         "pair_label":"flip" if pair["relation_flip"] else "hold"})
    assert len(rows)==128 and len({x["id"] for x in rows})==128
    assert len([x for x in rows if x["pair_label"]=="flip"])==64
    return rows

def prepare():
    data=HERE/"pilot_01"
    sealed=HERE/"cpu_matched_01"
    if sealed.exists():raise FileExistsError("Cannot overwrite CPU experiment")
    sealed.mkdir()
    train=build_signals.load_rows(data/"balanced_train_candidate.jsonl")
    assert len(train)==128
    assert all(x["split"]=="train" for x in train)
    assert Counter(x["label"] for x in train)=={0:64,1:64}
    batches={}; bypair={}
    for x in train:bypair.setdefault(x["curriculum_pair_id"],[]).append(x)
    assert len(bypair)==64 and all(len(v)==2 for v in bypair.values())
    pairs=list(bypair.values())
    # Build two mirrored batches per step: each contains one flip and one hold pair.
    flipped=[p for p in pairs if p[0]["curriculum_class"]=="flip"]
    held=[p for p in pairs if p[0]["curriculum_class"]=="hold"]
    assert len(flipped)==len(held)==32
    rng=random.Random(20261119)
    rng.shuffle(flipped); rng.shuffle(held)
    paired=[]
    for f,h in zip(flipped,held):
        row=f+h
        rng.shuffle(row)
        paired.append([r["id"] for r in row])
    standard=[rid for row in paired for rid in row]
    # Block randomizer: ensure the *same 128 examples* but break pair colocation.
    rng.shuffle(standard)
    shuffled=[standard[i:i+4] for i in range(0,128,4)]
    assert Counter(x for b in paired for x in b)==Counter(x for b in shuffled for x in b)
    # 32 fixed action replays from original TRAIN, identical for both arms.
    old=build_signals.load_rows(COMP/"training_17.jsonl")
    assert len(old)==120 and all(len(x["actor"])==4 for x in old[:32])
    trainlookup={x["id"]:x for x in train}
    steps={}
    for name,ordered in (("paired",paired),("shuffled",shuffled)):
        steps[name]=[{"step":i,"query_ids":ids,"actor_source_step":i,
                      "query_labels":[trainlookup[j]["label"] for j in ids]}
                    for i,ids in enumerate(ordered)]
        dump_jsonl(sealed/(name+"_steps.jsonl"),steps[name])
    dev=select_dev(data,train)
    dump_jsonl(sealed/"dev_panel.jsonl",dev)
    dump_jsonl(sealed/"train_panel.jsonl",train)
    # Commit freeze with source checkpoint and scripts hashes before model calls.
    base=COMP/"study"/"truth17"
    cpseal=json.loads((base/"CHECKPOINT_SEALED.json").read_text(encoding="utf-8"))
    sealedcp={n:sha(base/"adapter"/n) for n in cpseal["adapter_hashes"]}
    assert sealedcp==cpseal["adapter_hashes"]
    plan={"version":"kci-cpu-matched-01","seed":17,"arms":["paired","shuffled"],
          "updates_per_arm":32,"query_batch_size":4,"replay_actor_batch_size":4,
          "lr":0.0002,"optimizer":"AdamW","clip":1.0,"max_tokens":1024,
          "model_start_path":base.relative_to(ROOT).as_posix(),
          "model_start_seal":cpseal,"source_commit":"13c2997",
          "training_scope":"existing task query and actor replay; no new independent business tasks",
          "task_split":"source-adapted / developed synthetic grammar / dev not blind",
          "relevant_label_coverage":{"flip_pairs":32,"hold_pairs":32,
                                      "positive_queries":64,"negative_queries":64},
          "source_hashes":{"train":sha(data/"balanced_train_candidate.jsonl"),
                           "dev":sha(data/"queries.jsonl"),
                           "old_actor":sha(COMP/"training_17.jsonl")},
          "no_paid_api":True,"no_large_gpu":True,"study_status":"PREFLIGHT_ONLY",
          "train_ready_as_formal_dpo":False}
    # Record all step and dataset hashes; preflight and training scripts added to freeze in a separate manifest later.
    plan["frozen_data_hashes"]={p.name:sha(p) for p in sealed.glob("*.jsonl")}
    (sealed/"PLAN.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    print(json.dumps({"preparation":"PASS","steps":32,"train_rows":128,"dev_rows":128,
                      "model_start":plan["model_start_path"],"plan_sha":sha(sealed/"PLAN.json")}))
if __name__=="__main__":prepare()
