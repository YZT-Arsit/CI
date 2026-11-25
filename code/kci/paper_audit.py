"""Independent CPU replay of GPU-collected training evidence and failure attribution."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from .paper_alignment import Alignment, actor, metrics, prompt
from .paper_run import unpack
from .paper_worlds import FINAL, PhaseWorld, canon, filehash, key


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def saved(path):
    with gzip.open(path,"rt",encoding="utf8") as f:
        for line in f:yield json.loads(line)


def audit(packet,run):
    worlds={w.id:(s,w) for s,w in unpack(packet)}
    config=read(run/"config.json")
    if config["packet"]!=filehash(packet/"manifest.json"):raise ValueError("Wrong source packet")
    records={(r["world"],r["state"],r["remaining"]):r for r in saved(run/"fit_evidence.jsonl.gz")}
    decisions={r["observation_id"]:r["action"] for r in saved(run/"fit_policy.jsonl.gz")}
    def current(ob):
        actions=ob["legal_actions"]
        a=actions[0] if len(actions)==1 else decisions[key(prompt(ob))]
        assert a in actions
        return a
    native={};counts=Counter();statuses=Counter();traces={};classes=Counter()
    eligible_by_source=defaultdict(set);repairs=Counter()
    for tag,record in records.items():
        wid,sid,left=tag
        split,w=worlds[wid]
        assert split=="fit" and record["current_policy"]==config["initial"]
        assert record["reference_policy"]==w.reference_version
        if wid not in native:native[wid]=Alignment(w,current,config["initial"])
        eng=native[wid]
        s=type(w.initial())(**record["snapshot"])
        ob=w.observe(s)
        assert eng.sid(s)==sid and actor(ob)==record["actor"] and key(prompt(ob))==record["observation_id"]
        assert [r["action"] for r in record["candidates"]]==ob["legal_actions"]
        for row in record["candidates"]:
            a=row["action"];branches=w.branches(s,a)
            assert row["transitions"]==[dict(mass=str(m),after=eng.sid(n),audit=v) for m,n,v in branches]
            for _,n,v in branches:
                if v["verdict"] not in ("VALID","VIOLATION"):continue
                if isinstance(w,PhaseWorld):
                    direct=inferred=False
                    for recipient in s.delivered:
                        new=w.env.knowledge(n,recipient)-w.env.knowledge(s,recipient)
                        assert set(v["delta"].get(recipient,[]))==new
                        if not (w.env.authorized and recipient==w.env.spec.target):
                            direct |= any(ev["recipient"]==recipient and ev["fact"]==FINAL for ev in v["events"])
                            inferred |= FINAL in new
                else:
                    def joins(state):
                        return set(map(canon,w.env.recipient_knowledge(state)["inferred_protected_associations"]))
                    added=joins(n)-joins(s)
                    direct=False
                    inferred=bool(added and not n.authorized_relation)
                    assert set(map(canon,v["delta"][w.mod.RECIPIENT]["inferred_protected_associations"]))==added
                    if any(ev["status"]=="FAILED" for ev in v["events"]):
                        assert n.broker_receipts==s.broker_receipts and n.recipient_receipts==s.recipient_receipts
                        assert not added
                        counts["failed_delivery_branches_checked"]+=1
                assert (v["direct"],v["inferred"])==(direct,inferred)
            counts["candidate_transitions"]+=len(branches)
            classes[row["evidence_class"]]+=1
            verdicts={v["verdict"] for _,_,v in branches}
            assert row["immediate"]==(next(iter(verdicts)) if len(verdicts)==1 else "UNKNOWN")
            if row["immediate"]!="VALID":continue
            ref,witness=eng.follow(s,a,left,"reference")
            frozen,_=eng.follow(s,a,left,"current")
            assert metrics(frozen)==row["current"]
            assert metrics(ref)==row["reference"] and row["witness"]==witness
            if witness is not None:counts["safe_witnesses"]+=1
    for t in saved(run/"fit_traces.jsonl.gz"):
        if "failures" in t:continue
        wid=t["initial_world"];_,w=worlds[wid];eng=native[wid];s=w.initial()
        direct=infer=unknown=False
        for item in t["path"]:
            ob=w.observe(s)
            assert eng.sid(s)==item["state"] and item["observation_id"]==key(prompt(ob)) and item["actor"]==actor(ob)
            assert item["action"] in ob["legal_actions"]
            if t["policy"]=="current":assert item["action"]==current(ob)
            matches=[(n,v) for _,n,v in w.branches(s,item["action"]) if eng.sid(n)==item["after"]]
            assert len(matches)==1
            s,v=matches[0];assert v==item["audit"]
            direct|=v["direct"];infer|=v["inferred"];unknown|=v["verdict"] not in ("VALID","VIOLATION")
        done,success,cost,abort=w.terminal(s)
        assert t["direct"]==direct and t["inferred"]==infer and t["cost"]==cost
        assert t["unknown"]==((unknown or not done) and not (direct or infer))
        assert t["stsr"]==bool(done and success and not (unknown or direct or infer))
        traces[t["id"]]=t;counts["native_collection_traces"]+=1
    ids=set()
    for row in saved(run/"preferences.jsonl.gz"):
        record=records[row["world"],row["state"],row["remaining"]]
        assert row["current_policy"]==config["initial"] and row["root"]==record["root"]
        assert row["observation_id"]==record["observation_id"]==key(row["prompt"])
        assert row["reference_policy"]==record["reference_policy"] and row["actor"]==record["actor"]
        assert row["norm"]==record["norm"]
        r={canon(c["action"]):c for c in record["candidates"]}
        good=r[canon(row["chosen"])];bad=r[canon(row["rejected"])]
        assert good["witness"] is not None and good["immediate"]=="VALID" and good is not bad
        if bad["immediate"]=="VIOLATION":
            assert row["source"] in ("ci","repair")
        else:
            assert row["source"] in ("future","repair") and bad["immediate"]=="VALID"
            def rank(c):
                x=c["reference"];assert x["unknown"]==0
                return -x["any_ci"],x["stsr"],-(x["cost_given_safe_success"] or 0)
            assert rank(good)>rank(bad)
        if row["source"]=="repair":
            t=traces[row["trace_id"]];assert t["policy"]=="current" and not t["stsr"] and not t["unknown"]
            pos=next(i for i,h in enumerate(t["path"]) if h["state"]==row["state"])
            assert pos>0 and all(h["audit"]["verdict"]=="VALID" for h in t["path"][:pos])
            w=worlds[row["world"]][1]
            assert w.downstream(w.observe(type(w.initial())(**record["snapshot"])))
            assert t["path"][pos]["action"]==row["rejected"]
            repairs["immediate_ci" if bad["immediate"]=="VIOLATION" else "continuation"]+=1
        eligible_by_source[row["source"]].add((row["world"],row["state"],row["remaining"]))
        statuses[row["source"]]+=1;ids.add(key(row))
    opt=read(run/"optimizer.json")
    assert opt["pi0"]==config["initial"] and all(set(r["row_ids"])<=ids for r in opt["steps"])
    assert opt["updated"]==filehash(run/"shared_adapter/adapter_model.safetensors")
    proof=read(run/"complete.json")
    assert proof["rows"]==dict(statuses) and proof["reload_equal"] and proof["weights"]==opt["updated"]
    for name in ("before","after"):
        scores=read(run/(name+".json"))
        assert all(worlds[r["world"]][0]=="validation" for r in scores["worlds"])
    return dict(**counts,fit_decision_states=len(records),candidate_evidence_classes=dict(classes),
                states_with_preferences={k:len(v) for k,v in eligible_by_source.items()},
                repair_categories=dict(repairs),preference_counts=dict(statuses),
                checked_updates=len(opt["steps"]),adapter_sha256=opt["updated"])


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("packet",type=Path);p.add_argument("run",type=Path)
    a=p.parse_args();print(json.dumps(audit(a.packet,a.run),indent=2))

