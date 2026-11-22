"""Independent native replay of frozen real-7B trajectories and train CCE data."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import collections,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_exact_policy_continuation_cpu import environment_public
from kci.diverse_ci_branch_env import digest
from kci_pretrained_llm_train_forks import to_preflight
from kci_role_lora_dpo import validate_pairs

DATA=ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"
A=DATA/"qwen7b_A_all28_v1"
DEFAULT=DATA/"qwen7b_full_exploratory8_v1"
PROMPT=DATA/"qwen7b_full_deliberative8_v1"
TESTFORKS=(DATA/"qwen7b_B_forced_DG18_v1",
           DATA/"qwen7b_B_forced_DG19_v1")
TRAIN=DATA/"qwen7b_B_CCE_train8_v1"
SOURCE=DATA/"source_R_verified_positive_v1"
PROBE=DATA/"qwen7b_B_train_policy_probe_v1"

def load(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]
def world(name):
    return next(r for rows in GROUP.values() for r in rows if r["world"]==name)

def verify_full(ep):
    row=world(ep["world"])
    env=environment_public(row)
    s=env.initial()
    for item in ep["trace"]:
        assert digest(asdict(s))==item["state_sha256_before"],(ep["world"],item["step"])
        nxt,a=env.transition(s,item["action"])
        assert a["verdict"]==item["native_verdict"]
        assert digest(asdict(nxt))==item["state_sha256_after"]
        assert len(nxt.native.delivered["R"])-len(s.native.delivered["R"])==item["recipient_R_new_receipts"]
        s=nxt
    assert bool(s.native.done)==ep["completed"]
    assert bool(s.native.violation)==ep["CI_violation"]
    assert bool(s.native.success and not s.native.violation)==ep["safe_business_success"]
    return len(ep["trace"])

def verify_suffix(env,state,forced,result):
    start=state
    assert not start.native.done
    s,a=env.transition(start,forced)
    assert a["verdict"]=="VALID"
    for item in result["trace"][1:]:
        assert digest(asdict(s))==item["native_hash_before"]
        nxt,au=env.transition(s,item["action"])
        assert au["verdict"]==item["verdict"]
        s=nxt
    assert s.native.done
    assert bool(s.native.violation)==result["CI_violation"]
    assert bool(s.native.success and not s.native.violation)==result["safe_success"]
    return len(result["trace"])-1

def main():
    counters=collections.Counter()
    full_by_name={e["world"]:e for e in load(DEFAULT/"EPISODES.jsonl")}
    for folder in (DEFAULT,PROMPT):
        episodes=load(folder/"EPISODES.jsonl")
        assert len(episodes)==8
        for ep in episodes:
            counters["native_full_actions"]+=verify_full(ep)
            counters["native_full_episodes"]+=1
        assert all(not ep["safe_business_success"] and not ep["CI_violation"] for ep in episodes)
        assert all(any(t["phase"]=="preflight" and t["action"]=="SKIP_PRIVATE_CHANNEL"
                       for t in ep["trace"]) for ep in episodes)
    test_forks=[]
    for p in TESTFORKS:test_forks.extend(load(p/"RESULTS.jsonl"))
    assert len(test_forks)==8
    for item in test_forks:
        w=world(item["world"])
        e=environment_public(w)
        s=e.initial()
        for step in full_by_name[item["world"]]["trace"]:
            if step["phase"]=="preflight":break
            s,a=e.transition(s,step["action"])
            assert a["verdict"]=="VALID"
        assert digest(asdict(s))==item["fork_state_sha256"]
        for action,res in item["continuations"].items():
            counters["native_test_forced_steps"]+=verify_suffix(e,s,action,res)
            counters["test_fork_suffixes"]+=1
        assert item["continuations"]["OPEN_PRIVATE_CHANNEL"]["safe_success"]
        assert not item["continuations"]["SKIP_PRIVATE_CHANNEL"]["safe_success"]
    raw=load(TRAIN/"FORKS_NATIVE_TRAIN_ONLY.jsonl")
    pref=validate_pairs(TRAIN/"DPO_ROLE_B_TRAIN.jsonl","Coordinator")
    assert len(raw)==len(pref)==8
    for item in raw:
        assert item["split"]=="train"
        e,s,prefix=to_preflight(world(item["world"]),item["A_chosen_by_pretrained_LLM"])
        assert digest(asdict(s))==item["native_exact_B_preflight_hash"]
        assert digest(e.observation(s))==item["preflight_role_observation_hash"]
        for action,res in item["continuation_RESULTS"].items():
            counters["native_train_forced_steps"]+=verify_suffix(e,s,action,res)
            counters["train_fork_suffixes"]+=1
    for pref_item in pref:
        matching=next(v for v in raw if v["world"]==pref_item["world"])
        assert pref_item["state_hash"]==matching["native_exact_B_preflight_hash"]
        assert pref_item["chosen_action"]=="OPEN_PRIVATE_CHANNEL"
        assert pref_item["rejected_action"]=="SKIP_PRIVATE_CHANNEL"
    source=validate_pairs(SOURCE/"DPO_SOURCE_TRAIN.jsonl","SourceAgent")
    assert len(source)==100
    assert sum(r["chosen_action"]=="SEND_RAW_R" for r in source)==72
    assert sum(r["chosen_action"]=="VERIFY_PRIVATE_R" for r in source)==28
    observed=load(A/"EPISODES.jsonl")
    assert len(observed)==28
    assert sum(r["is_reference_optimal"] for r in observed if r["strict_reference_preference"])==16
    probe=json.loads((PROBE/"B_ACTIONS.json").read_text(encoding="utf8"))["summary"]
    assert probe["n_OPEN"]==0 and probe["n_SKIP"]==8
    counters["train_A_model_decisions"]=len(observed)
    counters["B_CCE_train_pairs"]=len(pref)
    counters["source_train_pairs"]=len(source)
    counters["source_legal_RAW"]=72
    counters["source_inferential_CI_violation"]=28
    counters["independent_train_roots"]=len(set(x["root"] for x in pref))
    out=DATA/"INDEPENDENT_NATIVE_EVIDENCE_AUDIT.json"
    report={"status":"PASS_ACTUAL_PRETRAINED_MODEL_NATIVE_TRACE_AND_CCE_REPLAY",
            "model":"qwen2.5:7b-instruct-q4_K_M",
            "train_test_split_separated":True,
            "developer_seen_eval_not_blind":True,
            "frozen_LLM_no_parameter_update":True,
            "source_R_positive_from_CPU_policy_not_Qwen":True,
            "counts":dict(counters)}
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("INDEPENDENT_REPLAY_PASS",json.dumps(report),flush=True)
if __name__=="__main__":main()
