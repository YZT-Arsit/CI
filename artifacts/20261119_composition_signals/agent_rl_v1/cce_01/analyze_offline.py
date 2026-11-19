"""Read-only paired diagnosis of backed-up terminal GRPO and CCE Q* runs.
No model forward, no GPU, no experiment changes.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,tarfile,statistics
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
RUNS={
 "terminal":(BASE/"grpo_64_completed_backup.tar.gz","gpu_4b_grpo_64_01"),
 "cce":(HERE/"cce_64_completed_backup.tar.gz",
        "artifacts/20261119_composition_signals/agent_rl_v1/gpu_4b_cce_64_01")
}
def fetch(tar,path):
    f=tar.extractfile(path)
    if f is None:raise FileNotFoundError(path)
    return f.read()
def records(data):
    return [json.loads(s) for s in data.decode().splitlines() if s.strip()]
def read_experiment(archive,prefix):
    archive=archive.resolve()
    if not archive.is_file():raise FileNotFoundError(archive)
    with tarfile.open(archive,"r:gz") as tar:
        names=[s.name for s in tar.getmembers()]
        if prefix+"/SUMMARY.json" not in names:
            raise FileNotFoundError("No exact prefix "+prefix+" in "+str(archive))
        md=json.loads(fetch(tar,prefix+"/MANIFEST.json"))
        needed=["RUN_CONFIG.json","groups.jsonl","SUMMARY.json",
                "dev_before/episodes.jsonl","dev_after/episodes.jsonl"]
        if "cce" in prefix:needed.append("cce_proofs.jsonl")
        files={}
        for n in needed:
            content=fetch(tar,prefix+"/"+n)
            if n in md:
                assert hashlib.sha256(content).hexdigest()==md[n],n
            files[n]=content
    return {
      "runconfig":json.loads(files["RUN_CONFIG.json"]),
      "summary":json.loads(files["SUMMARY.json"]),
      "groups":records(files["groups.jsonl"]),
      "before":records(files["dev_before/episodes.jsonl"]),
      "after":records(files["dev_after/episodes.jsonl"]),
      "proofs":records(files["cce_proofs.jsonl"]) if "cce_proofs.jsonl" in files else [],
      "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest()
    }
def map_episodes(ep):
    o={(e["scenario_key"],e["sample_index"]):e for e in ep}
    assert len(o)==len(ep)
    return o
def cmp_episodes(left,right):
    a,b=map_episodes(left),map_episodes(right)
    assert a.keys()==b.keys()
    same_actions=0
    same_outcome=0
    n=len(a)
    first_div=Counter()
    common_logp_abs=[]
    changed=[]
    for key in a:
        x,y=a[key],b[key]
        if (x["success"],x["violation"]) == (y["success"],y["violation"]):
            same_outcome+=1
        dx,dy=x["decisions"],y["decisions"]
        actx=[r["chosen_action"] for r in dx]
        acty=[r["chosen_action"] for r in dy]
        if actx==acty:same_actions+=1
        else:
            diverge=next((i for i,(u,v) in enumerate(zip(actx,acty)) if u!=v),
                         min(len(actx),len(acty)))
            first_div[diverge]+=1
            changed.append({"scenario_key":key[0],"sample_index":key[1],
              "left_action_path":actx,"right_action_path":acty,
              "left_success":x["success"],"right_success":y["success"],
              "left_violation":x["violation"],"right_violation":y["violation"]})
        for u,v in zip(dx,dy):
            if u["state_hash"]!=v["state_hash"] or u["chosen_action"]!=v["chosen_action"]:break
            common_logp_abs.append(abs(u["old_logprob"]-v["old_logprob"]))
    return {
      "n":n,"identical_action_paths":same_actions,
      "identical_end_outcomes":same_outcome,
      "changed_action_paths":n-same_actions,
      "changed_examples":changed[:20],
      "first_divergence_step_hist":dict(first_div),
      "common_actions_compared":len(common_logp_abs),
      "mean_abs_action_logp_change":statistics.mean(common_logp_abs) if common_logp_abs else None,
      "max_abs_action_logp_change":max(common_logp_abs) if common_logp_abs else None
    }
def main():
    data={k:read_experiment(*v) for k,v in RUNS.items()}
    root_fields=("model","seed","max_updates","group_size","dev_world_limit",
        "lr","clip_eps","kl_coef","explore_epsilon","temperature",
        "lora_rank","lora_alpha","bits4","dtype")
    args={x:{k:dat["runconfig"].get(k) for k in root_fields}
            for x,dat in data.items()}
    mismatch={k:(args["terminal"][k],args["cce"][k]) for k in root_fields
              if args["terminal"][k]!=args["cce"][k]}
    metrics={}
    for name,x in data.items():
        groups=x["groups"]
        vals=Counter()
        norm=[]
        kls=[]
        ratios=[]
        for g in groups:
            vals["updated"]+=bool(g["updated"])
            vals["skipped"]+=not g["updated"]
            vals[g["signal_kind"]]+=1
            if g["updated"]:
                norm.append(g["pre_clip_gradient_norm"])
                kls.append(g.get("mean_reference_kl",0))
                ratios.append(g.get("mean_ratio",0))
        metrics[name]={
           "before_safe_success":x["summary"]["before"]["safe_success"],
           "after_safe_success":x["summary"]["after"]["safe_success"],
           "before_violation":x["summary"]["before"]["violations"],
           "after_violation":x["summary"]["after"]["violations"],
           "group_counts":dict(vals),
           "avg_preclip_grad_norm":statistics.mean(norm) if norm else None,
           "max_preclip_grad_norm":max(norm) if norm else None,
           "avg_kl":statistics.mean(kls) if kls else None,
           "avg_ratio":statistics.mean(ratios) if ratios else None,
        }
    proofs=data["cce"]["proofs"]
    byrole=Counter()
    gaps=Counter()
    positive=Counter()
    for p in proofs:
        byrole[p["role"]]+=1
        positive["positive" if p["score"]>0 else "negative"]+=1
        v=[x["q"] for x in p["alternatives"].values() if x["legal"] and x is not None]
        span=max(v)-min(v) if len(v)>1 else 0
        if span<0.011:gaps["near_1_step"]+=1
        elif span<0.051:gaps["short_cost"]+=1
        elif span>.5:gaps["reachability_like"]+=1
        else:gaps["other"]+=1
    diag={
      "source":"Independent read-only archive diagnostics",
      "run_input_discrepancies":mismatch,
      "metrics":metrics,
      "original_baselines_identical_before_training":cmp_episodes(
          data["terminal"]["before"],data["cce"]["before"]),
      "terminal_before_after_action_changes":cmp_episodes(
          data["terminal"]["before"],data["terminal"]["after"]),
      "cce_before_after_action_changes":cmp_episodes(
          data["cce"]["before"],data["cce"]["after"]),
      "trained_policy_terminal_vs_cce":cmp_episodes(
          data["terminal"]["after"],data["cce"]["after"]),
      "cce_proofs":{"total":len(proofs),"roles":dict(byrole),
          "positive_negative":dict(positive),"relative_q_spread":dict(gaps)},
      "conclusions_not_supported":["causal mechanism proven","domain transfer",
            "frozen reference-policy value function"]
    }
    target=HERE/"OFFLINE_DIAGNOSTICS.json"
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(diag,indent=2,ensure_ascii=False),encoding="utf-8")
    brief={k:v for k,v in diag.items() if k not in (
        "original_baselines_identical_before_training",
        "terminal_before_after_action_changes","cce_before_after_action_changes",
        "trained_policy_terminal_vs_cce")}
    brief["baseline_action_control"]={
      "starts_equal":diag["original_baselines_identical_before_training"]["identical_action_paths"],
      "terminal_changed":diag["terminal_before_after_action_changes"]["changed_action_paths"],
      "cce_changed":diag["cce_before_after_action_changes"]["changed_action_paths"],
      "trained_final_disagree":diag["trained_policy_terminal_vs_cce"]["changed_action_paths"]
    }
    print(json.dumps(brief,ensure_ascii=False))
if __name__=="__main__":main()
