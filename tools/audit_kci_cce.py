"""Independent train-only CCE Q* credit replay/validation.

Recomputes counterfactual labels from ORIGINAL executable DynamicTask; does not
trust the training script's CCE annotations. Does not alter archived model files.
Use after the standard full native rollout/manifest auditor passes.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re,sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import Scenario,digest
from kci.agent_rl.cce import EARLY,EXCLUDED

PAT=re.compile(r"^(meeting|query)_related([01])_allowed([01])_h(\d+)$")
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def spec(key):
    m=PAT.fullmatch(key)
    if not m:raise ValueError(key)
    return Scenario(m[1],m[2]=="1",m[3]=="1",int(m[4]))
def raw_value(env,state,action):
    next_state,audit=env.transition(state,action)
    if audit["verdict"]=="UNKNOWN":raise AssertionError("UNKNOWN continuation")
    if audit["verdict"]=="VIOLATION":
        return {"legal":False,"reachable":False,"q":-1.,"optimal_terminal_turn":None}
    val=env.exact(next_state)
    if val["status"]=="PROVEN_UNREACHABLE":
        return {"legal":True,"reachable":False,"q":0.,"optimal_terminal_turn":None}
    assert val["status"]=="EXACT",val
    return {"legal":True,"reachable":True,
        "q":1.-0.01*int(val["min_total_cost"]),
        "optimal_terminal_turn":int(val["min_total_cost"])}
def run(out):
    man=read(out/"MANIFEST.json")
    for name,h in man.items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h,name
    cfg=read(out/"RUN_CONFIG.json")
    method=read(out/"CCE_CONFIG.json")
    assert cfg["credit_mode"]=="cce_oracle"
    min_gap=float(method["minimum_q_difference"])
    root=Path(cfg["source_root"])
    ps=rows(out/"cce_proofs.jsonl")
    episodes=rows(out/"episodes.jsonl")
    groups=rows(out/"groups.jsonl")
    count=sum(int(x["cce_proof_count"]) for x in groups)
    assert len(ps)==count
    keys=set()
    for p in ps:
        key=(p["training_group"],p["sample_index"],p["decision_index"])
        assert key not in keys
        keys.add(key)
    real=Counter();pos=Counter();idx=0;recomputed_cache={}
    for i,g in enumerate(groups):
        n=g["sample_size"]
        batch=episodes[idx:idx+n];idx+=n
        assert len(batch)==n and set(e["scenario_key"] for e in batch)=={g["group_id"]}
        s0=spec(g["group_id"])
        actual=[]
        for ep in batch:
            env=s0.build(root)
            state=env.decision_start()
            assert digest(asdict(state))==ep["initial_state_hash"]
            for step,d in enumerate(ep["decisions"]):
                assert digest(asdict(state))==d["state_hash"]
                obs=env.observation(state)
                assert digest(obs)==d["observation_hash"]
                assert d["chosen_action"] in env.options(state)
                if (d["actor"],obs["phase"]) in EARLY and len(env.options(state))>=2:
                    idkey=(i,ep["sample_index"],step)
                    stablekey=(s0.key,d["state_hash"])
                    if stablekey not in recomputed_cache:
                        recomputed_cache[stablekey]={
                           a:raw_value(env,state,a) for a in env.options(state)}
                    alts=recomputed_cache[stablekey]
                    legal={a:v["q"] for a,v in alts.items()
                           if v["legal"] and a not in EXCLUDED}
                    if len(legal)>=2:
                        span=max(legal.values())-min(legal.values())
                        if span>=min_gap and d["chosen_action"] in legal:
                            mean=sum(legal.values())/len(legal)
                            score=max(-1.,min(1.,2*(legal[d["chosen_action"]]-mean)/span))
                            if abs(score)>1e-8:
                                actual.append((idkey,d["state_hash"],score,alts))
                after,audit=env.transition(state,d["chosen_action"])
                assert audit["verdict"]==d["verdict"]
                state=after
            assert state.done and state.success==ep["success"] and state.violation==ep["violation"]
        pgroup=[p for p in ps if p["training_group"]==i]
        assert g["cce_proof_count"]==len(pgroup)==len(actual),(i,len(actual),len(pgroup))
        actualmap={x[0]:x for x in actual}
        for proof in pgroup:
            key=(proof["training_group"],proof["sample_index"],proof["decision_index"])
            assert key in actualmap,key
            _,statehash,score,alts=actualmap[key]
            assert proof["state_hash"]==statehash
            assert math.isclose(proof["score"],score,rel_tol=1e-8,abs_tol=1e-8)
            assert set(proof["alternatives"])==set(alts)
            for a,value in alts.items():
                expected=proof["alternatives"][a]
                assert expected["legal"]==value["legal"]
                assert expected["reachable"]==value["reachable"]
                assert expected["optimal_terminal_turn"]==value["optimal_terminal_turn"]
                assert math.isclose(expected["q"],value["q"],abs_tol=1e-8)
            real["local_credits"]+=1
            pos[proof["role"]]+=1
        real["groups"]+=1
        real["episodes"]+=n
        real["optimizer_updates"]+=int(g["updated"])
        real["outcome_contrast_groups"]+=g["signal_kind"]=="outcome_contrast"
        real["cost_only_groups"]+=g["signal_kind"]=="cost_only"
    assert idx==len(episodes) and real["local_credits"]==len(ps)
    result={"CCE_INDEPENDENT_ORACLE_REPLAY":"PASS",
        "groups":real["groups"],"episodes":real["episodes"],
        "local_proofs":len(ps),"role_credits":dict(pos),
        "model_optimizer_steps":real["optimizer_updates"],
        "all_evidence_hashed":True,
        "code_proves_training_only_native_Qstar_not_expected_policy_value":True,
        "source_business_families_independent":False,
        "cuda_transfer_not_required_for_audit":True}
    print(json.dumps(result,ensure_ascii=False))
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("experiment_dir",type=Path)
    a=p.parse_args()
    run(a.experiment_dir.resolve())
