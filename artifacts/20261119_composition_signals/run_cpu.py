"""KCI matched CPU LoRA diagnostic: frozen paired vs shuffled curriculum.

Same 128 training questions and 32 archived actor-replay batches, identical
source checkpoint and optimizer. Only query minibatch grouping differs.
No GPT API or GPU. Results on developed synthetic panel are not blind evidence.
"""
from __future__ import annotations
import gc,hashlib,json,os,re,sys,time
from collections import Counter,defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
COMP=ROOT/"artifacts"/"20261117_composition_repair"
sys.path.insert(0,str(HERE));sys.path.insert(0,str(COMP))
import build_signals
import run_repair as parent
w=parent.w
S=HERE/"cpu_matched_01"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return build_signals.load_rows(p)
def append(p,row):
    with p.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
def assert_plan():
    plan=read(S/"PLAN.json")
    pre=read(S/"TOKEN_PREFLIGHT.json")
    assert plan["updates_per_arm"]==32 and plan["arms"]==["paired","shuffled"]
    assert pre["preflight"]=="PASS" and pre["plan_hash"]==sha(S/"PLAN.json")
    for name,expected in plan["frozen_data_hashes"].items():
        assert sha(S/name)==expected,(name,"freeze altered")
    assert sha(HERE/"pilot_01"/"balanced_train_candidate.jsonl")==plan["source_hashes"]["train"]
    assert sha(HERE/"pilot_01"/"queries.jsonl")==plan["source_hashes"]["dev"]
    assert sha(COMP/"training_17.jsonl")==plan["source_hashes"]["old_actor"]
    cp=ROOT/plan["model_start_path"]
    assert read(cp/"CHECKPOINT_SEALED.json")==plan["model_start_seal"]
    assert all(sha(cp/"adapter"/n)==h for n,h in plan["model_start_seal"]["adapter_hashes"].items())
    return plan

def parse(raw,alphabet,elapsed):
    found=re.findall(r"ANSWER:\s*(["+alphabet+r"])\b",raw)
    return found[0] if len(found)==1 and 0<elapsed<9.5 else None

def eval_dev(tok,model,folder):
    items=rows(S/"dev_panel.jsonl");pred={}
    for start in range(0,len(items),4):
        batch=items[start:start+4]
        raw,secs=w.generate(tok,model,[r["prompt"] for r in batch])
        for r,txt in zip(batch,raw):
            output=parse(txt,"01",secs);pred[r["id"]]=output
            append(folder/"dev_outputs.jsonl",{"id":r["id"],"raw":txt,"output":output,"seconds_batch":secs})
    pairs=defaultdict(list)
    counter=Counter()
    role=defaultdict(Counter)
    for r in items:
        val=pred[r["id"]]
        counter["individual_total"]+=1;counter["individual_correct"]+=val==str(r["label"])
        counter["unknown"]+=val is None
        role[r["role"]]["individual_total"]+=1
        role[r["role"]]["individual_correct"]+=val==str(r["label"])
        pairs[r["pair_group"]].append(r)
    for group,values in pairs.items():
        assert len(values)==2 and len({r["pair_label"] for r in values})==1
        cls=values[0]["pair_label"]
        okay=all(pred[r["id"]]==str(r["label"]) for r in values)
        counter[cls+"_pairs"]+=1;counter[cls+"_both_correct"]+=int(okay)
        role[values[0]["role"]][cls+"_pairs"]+=1
        role[values[0]["role"]][cls+"_both_correct"]+=int(okay)
    assert counter["flip_pairs"]==32 and counter["hold_pairs"]==32
    return {"summary":dict(counter),"by_role":{k:dict(v) for k,v in sorted(role.items())}}

def eval_native(tok,model,folder):
    states=rows(COMP/"states.jsonl");queries=rows(COMP/"queries.jsonl");labels=rows(COMP/"labels.jsonl")
    assert (len(states),len(queries),len(labels))==(80,192,192)
    native=folder/"native";native.mkdir(exist_ok=False)
    parent.evaluate(tok,model,native,"direct",states,queries)
    qraw=rows(native/"direct_queries.jsonl");araw=rows(native/"direct_actions.jsonl")
    states_map={s["id"]:s for s in states}
    cases={l["state_id"]:l["case"] for l in labels}
    stats=Counter();byrole=defaultdict(Counter)
    for q,l,r in zip(queries,labels,qraw):
        assert q["id"]==l["id"]==r["id"]
        v=parse(r["raw"],"01",r["seconds_batch"])
        good=v==str(l["truth_target"])
        role=l["case"]["observation"]["role"]
        stats["query_correct"]+=int(good);stats["query_total"]+=1
        byrole[role]["query_correct"]+=int(good);byrole[role]["query_total"]+=1
    for s,r in zip(states,araw):
        assert s["id"]==r["id"]
        case=cases[s["id"]];cand={c["action"]:c for c in case["candidates"]}
        selected=r["selected"]
        best={case["candidates"][i]["action"] for i in case["optimal_indices"]}
        role=case["observation"]["role"]
        valid=r["valid"] and selected in cand
        stats["action_total"]+=1;stats["action_optimal"]+=int(valid and selected in best)
        stats["immediate_unsafe"]+=int(valid and not cand[selected]["immediate_safe"])
        stats["invalid_actions"]+=int(not valid)
        byrole[role]["action_total"]+=1;byrole[role]["action_optimal"]+=int(valid and selected in best)
        byrole[role]["immediate_unsafe"]+=int(valid and not cand[selected]["immediate_safe"])
    return {"summary":dict(stats),"by_role":{k:dict(v) for k,v in sorted(byrole.items())},
            "panel_scope":"Previously developed native fixed states; NOT a blind test or autonomous trajectory"}

def train_one(tok,model,folder,plan,arm):
    steps=rows(S/(arm+"_steps.jsonl"));all_old=rows(COMP/"training_17.jsonl")
    lookup={r["id"]:r for r in rows(S/"train_panel.jsonl")}
    opt=w.torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan["lr"],weight_decay=0.)
    letters=w.answer_ids(tok)
    for step,batch in enumerate(steps):
        start=time.perf_counter();model.train()
        ex=[lookup[i] for i in batch["query_ids"]]
        logits,ce,qt=w.forward_targets(tok,model,[x["prompt"] for x in ex],
            ["ANSWER:\n"+str(x["label"]) for x in ex])
        bit=ce[:,-2].mean()
        fmt=.1*w.torch.cat([ce[:,:-2],ce[:,-1:]],dim=1).mean()
        actor=all_old[step]["actor"]
        alogits,ace,at=w.forward_targets(tok,model,[x["prompt"] for x in actor],[x["answer"] for x in actor])
        abit,afmt=w.set_loss(alogits,ace,[letters[x["good_indices"]] for x in actor])
        loss=bit+fmt+abit+afmt
        opt.zero_grad();loss.backward()
        norm=float(w.torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        opt.step()
        elapsed=time.perf_counter()-start
        assert elapsed<30,("CPU budget exceeded",arm,step,elapsed)
        append(folder/"training.jsonl",{"step":step,"query_ids":batch["query_ids"],
            "labels":batch["query_labels"],"actor_source_step":step,
            "optimizer_step":step+1,"loss":float(loss.detach()),
            "bit_loss":float(bit.detach()),"format_loss":float(fmt.detach()),
            "actor_loss":float(abit.detach()),"actor_format_loss":float(afmt.detach()),
            "gradient_norm":norm,"query_tokens":qt,"actor_tokens":at,
            "seconds":elapsed})
        if (step+1)%8==0:print(json.dumps({"arm":arm,"step":step+1,"elapsed_step":round(elapsed,2)}),flush=True)
        del logits,ce,alogits,ace,bit,fmt,abit,afmt,loss
    model.save_pretrained(folder/"adapter")
    (folder/"CHECKPOINT_SEALED.json").write_text(json.dumps({
        "initial_checkpoint":plan["model_start_path"],
        "initial_seal":plan["model_start_seal"],
        "steps":32,
        "adapter_hashes":{x.name:sha(x) for x in (folder/"adapter").iterdir() if x.is_file()}
    },indent=2),encoding="utf-8")
    del opt

def run():
    plan=assert_plan()
    assert not (S/"study").exists(),"Study already exists; refusing overwrite"
    w.verify_model()
    (S/"study").mkdir()
    cp={"path":plan["model_start_path"],"seal":plan["model_start_seal"]}
    full={}
    for arm in ["initial","paired","shuffled"]:
        fold=S/"study"/arm;fold.mkdir()
        tok,model=w.load("start",{"checkpoints":{"start":cp}})
        initial=w.digest(model)
        assert initial==read(ROOT/plan["model_start_path"]/"EVAL_IDENTITY.json")["before"]
        (fold/"INITIAL.json").write_text(json.dumps({"digest":initial,"checkpoint":cp}),encoding="utf-8")
        if arm!="initial":
            train_one(tok,model,fold,plan,arm)
            saved=w.digest(model)
            # Validate persisted adapter bytes by reloading before evaluation.
            newseal=read(fold/"CHECKPOINT_SEALED.json")
            for filename,h in newseal["adapter_hashes"].items(): assert sha(fold/"adapter"/filename)==h
            del model;gc.collect()
            newcp={"path":fold.relative_to(ROOT).as_posix(),"seal":newseal}
            tok,model=w.load("start",{"checkpoints":{"start":newcp}})
            assert w.digest(model)==saved
        else:
            saved=initial
        model.eval();before=w.digest(model)
        dev=eval_dev(tok,model,fold)
        native=eval_native(tok,model,fold)
        assert before==w.digest(model)
        (fold/"IDENTITY.json").write_text(json.dumps({"before":before,"after":w.digest(model),
                             "initial":initial,"checkpoint_type":arm}),encoding="utf-8")
        full[arm]={"dev":dev,"native":native,"identity":{"initial":initial,"after":before}}
        (fold/"RESULTS.json").write_text(json.dumps(full[arm],indent=2),encoding="utf-8")
        print(json.dumps({"arm":arm,"dev":dev["summary"],"native":native["summary"]}),flush=True)
        del model;gc.collect()
    (S/"RESULTS.json").write_text(json.dumps({"arms":full,
            "comparison_scope":"Matched CPU training; synthetic-developed DEV and native regression; no causality beyond batch organization",
            "formal_method_contribution_established":False,"gpu_training":"NO_GO"},indent=2),encoding="utf-8")
    (S/"SEALED.json").write_text(json.dumps({"complete":True,"plan_hash":sha(S/"PLAN.json"),
        "training_code_hash":sha(HERE/"run_cpu.py"),
        "files":{p.relative_to(S).as_posix():sha(p) for p in (S/"study").rglob("*") if p.is_file()},
        "results_sha256":sha(S/"RESULTS.json")},indent=2),encoding="utf-8")
    print(json.dumps({"study":"COMPLETE","arms":list(full),"gpu":"NO_GO"}),flush=True)
if __name__=="__main__":run()
