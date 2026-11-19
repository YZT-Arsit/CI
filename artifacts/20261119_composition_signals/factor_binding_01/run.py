"""ONE bounded KCI factor-binding comparison: task-replay control vs learned relation.

No API or GPU. Same start, task examples, replay and optimizer updates.
Uses model SELF-GENERATED relation estimates at inference, never true oracle notes.
"""
from __future__ import annotations
import gc, hashlib, json, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent;ROOT=BASE.parents[1]
COMP=ROOT/"artifacts"/"20261117_composition_repair"
sys.path[:0]=[str(BASE),str(HERE),str(COMP),str(ROOT/"artifacts"/"20261117_relation_bridge")]
import run_cpu as prior
import build_signals
from prepare import relation_query
from repair_data import note
from bridge_data import relation_fact
w=prior.w
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return build_signals.load_rows(p)
def append(p,v):
    with p.open("a",encoding="utf-8") as f:f.write(json.dumps(v,ensure_ascii=False)+"\n")
def parse(text,alphabet,seconds):
    m=re.findall(r"ANSWER:\s*(["+alphabet+r"])\b",text)
    return m[0] if len(m)==1 and 0<seconds<9.5 else None
def verify():
    plan=read(HERE/"PLAN.json");pre=read(HERE/"TOKEN_PREFLIGHT.json")
    assert pre["preflight"]=="PASS" and pre["plan_sha256"]==sha(HERE/"PLAN.json")
    assert plan["cpu_training_seed"]==17 and plan["updates_per_arm"]==64
    for name,h in plan["data_sha256"].items():assert sha(HERE/(name+".jsonl"))==h
    cp=ROOT/plan["start_checkpoint"]
    assert read(cp/"CHECKPOINT_SEALED.json")==plan["start_seal"]
    assert all(sha(cp/"adapter"/n)==h for n,h in plan["start_seal"]["adapter_hashes"].items())
    assert not (HERE/"study").exists(),"Refusing repeated model experiment"
    # Verify old frozen task-query text before any model calls.
    native=rows(COMP/"queries.jsonl")
    assert len(native)==192
    for x in native: relation_query(x["prompts"]["direct"])
    return plan

def eval_questions(tok,model,panel,folder):
    assert len({r["id"] for r in panel})==len(panel)
    results={"task":{},"relation":{}}
    for mode in ("task","relation"):
        for start in range(0,len(panel),4):
            batch=panel[start:start+4]
            prompts=[r["task_query"] if mode=="task" else r["relation_query"] for r in batch]
            raw,secs=w.generate(tok,model,prompts)
            for r,v in zip(batch,raw):
                pred=parse(v,"01",secs)
                results[mode][r["id"]]=pred
                append(folder/(mode+"_outputs.jsonl"),{"id":r["id"],"raw":v,"estimate":pred,
                            "seconds_batch":secs})
    scores={}
    for mode,target in (("task","task_target"),("relation","relation_target")):
        stat=Counter();roles=defaultdict(Counter)
        for r in panel:
            good=results[mode][r["id"]]==str(r[target])
            stat["n"]+=1;stat["correct"]+=good;stat["invalid"]+=results[mode][r["id"]] is None
            roles[r["role"]]["n"]+=1;roles[r["role"]]["correct"]+=good
        scores[mode]={"totals":dict(stat),"by_role":{k:dict(v) for k,v in roles.items()}}
    pairing=defaultdict(list)
    for r in panel:
        # Native task world has 2 sides under group; each state/candidate is stable
        group=(r["base_group_id"],r["role"],r["action"],r["id"].split(":")[-2])
        pairing[group].append(r)
    ps=Counter();rps=Counter()
    for group,records in pairing.items():
        assert len(records)==2,("Expected paired same-root histories",group,len(records))
        for mode,target,summary in (("task","task_target",ps),("relation","relation_target",rps)):
            kind="flip" if records[0][target]!=records[1][target] else "hold"
            complete=all(results[mode][r["id"]]==str(r[target]) for r in records)
            summary[kind+"_pairs"]+=1;summary[kind+"_both_correct"]+=int(complete)
    scores["task_pairs"]=dict(ps)
    scores["relation_pairs"]=dict(rps)
    return scores

def native_relation(tok,model,folder,labels,queries):
    relation={}
    for i in range(0,len(queries),4):
        batch=queries[i:i+4]
        t=[relation_query(q["prompts"]["direct"]) for q in batch]
        raw,secs=w.generate(tok,model,t)
        for q,l,v in zip(batch,labels[i:i+4],raw):
            fact=relation_fact(l["case"]["observation"],q["action"])
            assert fact in (0,1)
            pred=parse(v,"01",secs)
            relation[q["id"]]=pred
            append(folder/"relation_outputs.jsonl",{"id":q["id"],"raw":v,
                   "estimate":pred,"true_relation":fact,"seconds_batch":secs})
    assert len(relation)==192
    return relation

def score_native_files(queries,states,labels,folder):
    qr=rows(folder/"direct_queries.jsonl")
    ar=rows(folder/"direct_actions.jsonl")
    assert len(qr)==len(queries)==192 and len(ar)==len(states)==80
    labels_by_state={l["state_id"]:l["case"] for l in labels}
    stats=Counter();role_stats=defaultdict(Counter)
    for q,l,r in zip(queries,labels,qr):
        assert q["id"]==l["id"]==r["id"]
        val=parse(r["raw"],"01",r["seconds_batch"])
        valid=val==str(l["truth_target"])
        role=l["case"]["observation"]["role"]
        stats["query_total"]+=1;stats["query_correct"]+=int(valid)
        role_stats[role]["query_total"]+=1;role_stats[role]["query_correct"]+=int(valid)
    for s,r in zip(states,ar):
        assert s["id"]==r["id"]
        case=labels_by_state[s["id"]]
        cand={c["action"]:c for c in case["candidates"]}
        chosen=r["selected"]
        good={case["candidates"][i]["action"] for i in case["optimal_indices"]}
        valid=r["valid"] and chosen in cand
        ok=valid and chosen in good
        unsafe=valid and not cand[chosen]["immediate_safe"]
        role=case["observation"]["role"]
        stats["action_total"]+=1;stats["action_optimal"]+=int(ok)
        stats["immediate_unsafe"]+=int(unsafe)
        stats["invalid_actions"]+=int(not valid)
        role_stats[role]["action_total"]+=1;role_stats[role]["action_optimal"]+=int(ok)
        role_stats[role]["immediate_unsafe"]+=int(unsafe)
    return {"totals":dict(stats),"by_role":{k:dict(v) for k,v in role_stats.items()}}

def eval_native_self(tok,model,folder):
    states=rows(COMP/"states.jsonl")
    queries=rows(COMP/"queries.jsonl")
    labels=rows(COMP/"labels.jsonl")
    assert (len(states),len(queries),len(labels))==(80,192,192)
    direct=prior.eval_native(tok,model,folder)
    if direct["summary"]["action_total"]!=80:raise AssertionError("Native direct missing")
    selfdir=folder/"self_fact";selfdir.mkdir()
    predicted=native_relation(tok,model,selfdir,labels,queries)
    composed=[]
    for q in queries:
        text=q["prompts"]["direct"]+note(predicted[q["id"]] if predicted[q["id"]] is not None else "unknown")
        composed.append({**q,"prompts":{"direct":text}})
    prior.parent.evaluate(tok,model,selfdir,"direct",states,composed)
    selfscore=score_native_files(queries,states,labels,selfdir)
    relstats=Counter()
    for q,l in zip(queries,labels):
        predicted_relation=predicted[q["id"]]
        factual=relation_fact(l["case"]["observation"],q["action"])
        relstats["total"]+=1;relstats["correct"]+=predicted_relation==str(factual)
        relstats["invalid"]+=predicted_relation is None
    return {"direct":direct,"self_generated_factor":selfscore,"relation":dict(relstats),
            "scope":"Native fixed developed states; external scorer ONLY, all model input relation facts are self generated."}

def train(tok,model,folder,plan,arm):
    dataset={r["id"]:r for r in rows(HERE/"train.jsonl")}
    schedule=rows(HERE/"schedule.jsonl")
    actor_source=rows(COMP/"training_17.jsonl")
    opt=w.torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan["learning_rate"],
                             weight_decay=plan["weight_decay"])
    letters=w.answer_ids(tok)
    for step,rec in enumerate(schedule):
        begin=time.perf_counter();model.train()
        ex=[dataset[n] for n in rec["query_ids"]]
        task_logits,task_ce,task_tokens=w.forward_targets(tok,model,[r["task_query"] for r in ex],
                                                         ["ANSWER:\n"+str(r["task_target"]) for r in ex])
        task_bit=task_ce[:,-2].mean()
        task_fmt=.1*w.torch.cat([task_ce[:,:-2],task_ce[:,-1:]],dim=1).mean()
        secondary=[r["task_query"] if arm=="task_replay_control" else r["relation_query"] for r in ex]
        labels=[r["task_target"] if arm=="task_replay_control" else r["relation_target"] for r in ex]
        extra_logits,extra_ce,extra_tokens=w.forward_targets(tok,model,secondary,["ANSWER:\n"+str(v) for v in labels])
        extra_bit=extra_ce[:,-2].mean()
        extra_fmt=.1*w.torch.cat([extra_ce[:,:-2],extra_ce[:,-1:]],dim=1).mean()
        acts=actor_source[rec["actor_source_step"]]["actor"]
        alogits,ace,atokens=w.forward_targets(tok,model,[a["prompt"] for a in acts],[a["answer"] for a in acts])
        actor_bit,actor_fmt=w.set_loss(alogits,ace,[letters[a["good_indices"]] for a in acts])
        loss=task_bit+task_fmt+extra_bit+extra_fmt+actor_bit+actor_fmt
        opt.zero_grad();loss.backward()
        grad=float(w.torch.nn.utils.clip_grad_norm_(model.parameters(),plan["gradient_clip"],
                                                   error_if_nonfinite=True))
        opt.step()
        seconds=time.perf_counter()-begin
        assert seconds<30,("CPU watchdog failed",arm,step,seconds)
        assert all(float(x.detach()).is_integer() and int(x.item())==step+1 for x in
                   [state["step"] for state in opt.state.values() if "step" in state])
        append(folder/"training.jsonl",{"step":step,"examples":rec["query_ids"],
            "actor_source_step":rec["actor_source_step"],"task_tokens":task_tokens,
            "extra_tokens":extra_tokens,"actor_tokens":atokens,
            "loss":float(loss.detach()),"task_bit":float(task_bit.detach()),
            "extra_bit":float(extra_bit.detach()),"actor_bit":float(actor_bit.detach()),
            "gradient_norm":grad,"seconds":seconds})
        if (step+1)%16==0:print(json.dumps({"arm":arm,"updates":step+1,"seconds_last":round(seconds,3)}),flush=True)
        del task_logits,task_ce,extra_logits,extra_ce,alogits,ace,loss
    model.save_pretrained(folder/"adapter")
    seal={"steps":64,"start_path":plan["start_checkpoint"],"initial_adapter_hashes":plan["start_seal"]["adapter_hashes"],
          "adapter_hashes":{f.name:sha(f) for f in (folder/"adapter").iterdir() if f.is_file()}}
    (folder/"CHECKPOINT_SEALED.json").write_text(json.dumps(seal,indent=2),encoding="utf-8")
    del opt

def run():
    plan=verify()
    w.verify_model()
    study=HERE/"study";study.mkdir(exist_ok=False)
    checkpoint={"path":plan["start_checkpoint"],"seal":plan["start_seal"]}
    manifest={}
    for arm in ("initial",)+tuple(plan["arms"]):
        folder=study/arm;folder.mkdir()
        tok,model=w.load("start",{"checkpoints":{"start":checkpoint}})
        initial=w.digest(model)
        expected=read(ROOT/plan["start_checkpoint"]/"EVAL_IDENTITY.json")["before"]
        assert initial==expected
        (folder/"INITIAL.json").write_text(json.dumps({"start":plan["start_checkpoint"],"digest":initial}),encoding="utf-8")
        if arm!="initial":
            train(tok,model,folder,plan,arm)
            saved=w.digest(model)
            seal=read(folder/"CHECKPOINT_SEALED.json")
            new_cp={"path":folder.relative_to(ROOT).as_posix(),"seal":seal}
            del model;gc.collect()
            tok,model=w.load("start",{"checkpoints":{"start":new_cp}})
            assert saved==w.digest(model),"Reloaded adapter parameters differ"
        model.eval()
        before=w.digest(model)
        for name in ("dev","confirm_depth4"):
            dpath=folder/name;dpath.mkdir()
            score=eval_questions(tok,model,rows(HERE/(name+".jsonl")),dpath)
            (dpath/"RESULTS.json").write_text(json.dumps(score,indent=2),encoding="utf-8")
            print(json.dumps({"arm":arm,"panel":name,"task":score["task"]["totals"],
               "relation":score["relation"]["totals"],"task_pairs":score["task_pairs"]}),flush=True)
        native=eval_native_self(tok,model,folder)
        assert before==w.digest(model),"Evaluation changed model weights"
        (folder/"IDENTITY.json").write_text(json.dumps({"initial":initial,"evaluated":before,
            "after":w.digest(model),"arm":arm}),encoding="utf-8")
        (folder/"NATIVE_RESULTS.json").write_text(json.dumps(native,indent=2),encoding="utf-8")
        print(json.dumps({"arm":arm,"native_direct":native["direct"]["summary"],
          "native_self":native["self_generated_factor"]["totals"],
          "native_relation":native["relation"]}),flush=True)
        del model;gc.collect()
    seal={"complete":True,"plan_hash":sha(HERE/"PLAN.json"),
          "training_script_hash":sha(HERE/"run.py"),
          "files":{p.relative_to(HERE).as_posix():sha(p) for p in study.rglob("*") if p.is_file()}}
    (HERE/"SEALED.json").write_text(json.dumps(seal,indent=2),encoding="utf-8")
    print(json.dumps({"study":"COMPLETE","arms":["initial",*plan["arms"]],"gpu":"NO_GO"}),flush=True)
if __name__=="__main__":run()
