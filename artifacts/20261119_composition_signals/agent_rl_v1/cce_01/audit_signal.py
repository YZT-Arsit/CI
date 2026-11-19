"""Read-only, no-GPU CCE opportunity audit on archived 4B on-policy trajectories.
Replay each visited prefix; enumerate legal counterfactual actions and exact
native continuation. Q* means oracle feasibility, NOT policy-expected return.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json,re,sys,tarfile
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"code"))
from kci.dynamic_tasks import canonical
from kci.agent_rl.core import Scenario,digest
HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parent/"grpo_64_completed_backup.tar.gz"
ENTRY="gpu_4b_grpo_64_01/"
PAT=re.compile(r"^(meeting|query)_related([01])_allowed([01])_h(\d+)$")
def sha(path):return sha256(path.read_bytes()).hexdigest()
def parse(b):return [json.loads(x) for x in b.decode().splitlines() if x.strip()]
def load():
    with tarfile.open(BUNDLE,"r:gz") as z:
        files={}
        for name in ("MANIFEST.json","episodes.jsonl","groups.jsonl"):
            files[name]=z.extractfile(ENTRY+name).read()
    manifest=json.loads(files["MANIFEST.json"])
    for n in ("episodes.jsonl","groups.jsonl"):
        assert sha256(files[n]).hexdigest()==manifest[n],n
    return parse(files["episodes.jsonl"]),parse(files["groups.jsonl"])
def scenario(s):
    m=PAT.fullmatch(s)
    if not m:raise ValueError(s)
    return Scenario(m[1],m[2]=="1",m[3]=="1",int(m[4]))
def qvalue(env,state,action):
    after,audit=env.transition(state,action)
    if audit["verdict"]=="UNKNOWN":raise AssertionError("Invalid contract")
    if audit["verdict"]=="VIOLATION":
        return {"valid":False,"reachable":False,"q":-1.0,"turns":None}
    exact=env.exact(after)
    if exact["status"]=="PROVEN_UNREACHABLE":
        return {"valid":True,"reachable":False,"q":0.0,"turns":None}
    assert exact["status"]=="EXACT"
    return {"valid":True,"reachable":True,
            "q":round(1-0.01*exact["min_total_cost"],5),
            "turns":exact["min_total_cost"]}
def main():
    out=HERE/"pilot"
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    episodes,groups=load()
    assert len(episodes)==512 and len(groups)==64
    unique={};counts=Counter()
    for e in episodes:
        env=scenario(e["scenario_key"]).build()
        s=env.decision_start()
        assert digest(asdict(s))==e["initial_state_hash"]
        for d in e["decisions"]:
            assert digest(asdict(s))==d["state_hash"]
            assert d["chosen_action"] in env.options(s)
            counts["steps"]+=1
            if len(env.options(s))>1:
                key=e["scenario_key"]+":"+d["state_hash"]
                if key not in unique:
                    options={a:qvalue(env,s,a) for a in env.options(s)}
                    legal=[x for x in options.values() if x["valid"]]
                    reach=len(legal)>1 and len({x["reachable"] for x in legal})>1
                    difference=len(legal)>1 and len({x["q"] for x in legal})>1
                    unique[key]={"state_key":key,"scenario":e["scenario_key"],
                       "role":d["actor"],"phase":env.observation(s)["phase"],
                       "options":options,"legal_options":len(legal),
                       "reachability_gap":reach,"any_future_gap":difference,
                       "visit_count":0}
                unique[key]["visit_count"]+=1
            nxt,audit=env.transition(s,d["chosen_action"])
            assert d["verdict"]==audit["verdict"]
            s=nxt
        assert s.done and s.success==e["success"] and s.violation==e["violation"]
    cells=defaultdict(Counter)
    for item in unique.values():
        category=("reachable_gap" if item["reachability_gap"] else
                  "cost_gap" if item["any_future_gap"] else
                  "equal_legal" if item["legal_options"]>1 else "one_legal")
        counts[category]+=1
        cells[(item["role"],item["phase"])][category]+=1
    (out/"prefixes.jsonl").write_text(
       "".join(canonical(x)+"\n" for x in unique.values()),encoding="utf-8")
    result={"source":"verified 64-group Qwen 4B 512 sampled episodes",
      "archive_sha256":sha(BUNDLE),"groups":len(groups),"episodes":len(episodes),
      "steps":counts["steps"],"distinct_branching_prefixes":len(unique),
      "signal_categories":{k:v for k,v in counts.items() if k!="steps"},
      "role_phase":{str(k):dict(v) for k,v in cells.items()},
      "oracle": "exact native task reachability Q*, training-only, NOT policy continuation",
      "no_model_training":True,"no_claim_of_business_generalization":True}
    (out/"SUMMARY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="role_phase"},ensure_ascii=False))
if __name__=="__main__":main()
