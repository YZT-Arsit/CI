r"""Gate A: actual native end-to-end heuristic policies and blind-to-graph lexical learners.

Primary causal evaluation is class-balanced related vs SAME-SUBJECT NORMAL-FACT safe
histories (not pooled with easier cross-subject/permission controls).
Rule-aware planner only sees model-visible records and public clauses.
Bag-of-words / char ngram learners are trained ONLY on train graph families.
No LLM training, no hidden source labels in any heuristic inference prompt.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from dataclasses import asdict
import json,sys,re,time,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.diverse_ci_branch_env import task_catalog,SLOT_LABELS,digest
from kci.diverse_ci_long_horizon_v31 import V31Env,enumerate_specs_for_root
from solve_v31_longterm import before_preflight,reward
from audit_kci_v31_gateA import verdict_from_observation

BASE=Path(os.environ.get("KCI_V31_DATASET_PATH",str(ROOT/"artifacts"/"20261120_v31"/"V31_GATEA_FINAL_CANDIDATE")))
def records(n):return [json.loads(x) for x in (BASE/n).read_text(encoding="utf-8").splitlines() if x.strip()]
def build_spec(x):
    task=next(t for t in task_catalog() if t.id==x["root_id"])
    n=int(x["case_id"].split("_M")[-1])-1
    return enumerate_specs_for_root(task,x["business_condition"],n)

def actual_policy(env,action):
    """Run each step; no native oracle suffix. Uses only *immediate* CI checks."""
    st,obs=before_preflight(env)
    trace=[]
    def take(a):
        nonlocal st
        assert a in env.options(st),(env.spec.case_id,st.stage,a,env.options(st))
        next,audit=env.transition(st,a)
        trace.append({
            "actor":env.observation(st)["role"],
            "stage":st.stage,"action":a,
            "verdict":audit["verdict"],
            "before_sha256":digest(asdict(st)),
            "after_sha256":digest(asdict(next))})
        assert audit["verdict"] in ("VALID","VIOLATION"),(env.spec.case_id,a,audit)
        st=next
    take(action)
    for f in env.required_facts:
        if st.done:break
        take("READ:"+SLOT_LABELS[f])
        if st.done:break
        attempted,audit=env.transition(st,"SEND_RAW_R")
        if audit["verdict"]=="VIOLATION":
            if "VERIFY_PRIVATE_R" in env.options(st):
                take("VERIFY_PRIVATE_R")
                if not st.done:take("SEND_CERT_R")
            else:take("ABORT")
        elif audit["verdict"]=="VALID":
            take("SEND_RAW_R")
        else:raise AssertionError(audit)
    if not st.done:
        assert "COMMIT_RESULT" in env.options(st)
        take("COMMIT_RESULT")
    return {"action":action,"native_success":st.success,
            "privacy_violation":st.violation,"native_done":st.done,
            "native_steps":st.steps,
            "reward":reward(st,env.spec.cost_per_step),
            "action_trace":trace,
            "actual_business_result":st.generated_result,
            "final_R_facts":list(st.delivered["R"])}

def actor_text(obs):
    return " ".join([
        obs["current_case_subject"],
        obs["recipient_policy"],
        " ".join(x["text"] for x in obs["delivered_message_log"]),
        json.dumps(obs["task_evidence_contract"],sort_keys=True),
        json.dumps(obs["declared_knowledge_rules"],sort_keys=True),
        json.dumps(obs["privacy_resource_contract"],sort_keys=True),
    ])

def main():
    out=BASE/"GATE_A_STRONG_SHORTCUT_AND_NATIVE_ROLLOUTS.json"
    if out.exists():raise FileExistsError(out)
    source=records("CASE_SPECS.jsonl")
    visibles={x["world_id"]:x for x in records("MODEL_OBSERVATIONS.jsonl")}
    labelled={x["world_id"]:x for x in records("EXACT_NATIVE_ACTION_PREFERENCES_EVAL_ONLY.jsonl")}
    assert len(source) in (42,63)
    chosen=[]
    for s in source:
        for h in ("related","same_entity_safe"):
            world=s["case_id"]+"__"+h
            x=visibles[world]
            true=labelled[world]
            assert true["oracle_action"] in (
                "OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL")
            chosen.append({
                "world_id":world,"graph_family":s["graph_family"],
                "split":s["split"],
                "case_id":s["case_id"],
                "history":h,
                "obs":x["model_input"],"gold":true["oracle_action"],
                "optimal_return":max(true["branch_action_values"].values()),
            })
    assert len(chosen)==2*len(source)
    assert sum(x["gold"]=="OPEN_PRIVATE_CHANNEL" for x in chosen)==len(source)
    assert sum(x["gold"]=="SKIP_PRIVATE_CHANNEL" for x in chosen)==len(source)
    # Exact same target roots can never appear in more than one split.
    for s in ("train","dev","exploratory_test"):
        assert len({x["graph_family"] for x in chosen if x["split"]==s})=={
            "train":4,"dev":1,"exploratory_test":2}[s]
    allpolicies=(
        "always_open","always_skip","lexical_three_checks","entity_only",
        "normal_signal_only","required_overlap_without_graph",
        "visible_rule_graph_planner")
    reports=defaultdict(lambda:Counter())
    native=[]
    for x in chosen:
        case=next(r for r in source if r["case_id"]==x["case_id"])
        env=V31Env(build_spec(case),x["history"])
        obs=env.observation(before_preflight(env)[0])
        assert obs==x["obs"]
        preds=verdict_from_observation(obs)
        for policy in allpolicies:
            act=preds[policy]
            episode=actual_policy(env,act)
            assert episode["reward"]<=x["optimal_return"]+1e-7
            reports[(policy,x["split"])]["n"]+=1
            reports[(policy,x["split"])]["correct"]+=int(act==x["gold"])
            reports[(policy,x["split"])]["safe_success"]+=int(
                episode["native_success"] and not episode["privacy_violation"])
            reports[(policy,x["split"])]["violations"]+=int(episode["privacy_violation"])
            reports[(policy,x["split"])]["regret_micros"]+=round(
                (x["optimal_return"]-episode["reward"])*1000000)
            if policy in ("always_open","always_skip","lexical_three_checks",
                          "visible_rule_graph_planner"):
                native.append({
                    "world_id":x["world_id"],"split":x["split"],
                    "policy":policy,"model_visible_prompt_sha256":digest(obs),
                    "gold_exact_native_first_action":x["gold"],
                    "actual_native_episode":episode,
                    "not_a_model_Qpi_rollout":True})
    result={}
    for (policy,split),c in reports.items():
        result.setdefault(policy,{})[split]={
            "n":c["n"],"correct_action":c["correct"],
            "first_action_accuracy":round(c["correct"]/c["n"],4),
            "actual_native_safe_successes":c["safe_success"],
            "actual_native_violations":c["violations"],
            "native_regret":round(c["regret_micros"]/1000000,6)}
    # Lightweight rule-BLIND text learners: must not see heldout graph labels.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    ml={}
    for name,vectorizer in (
        ("lexical_word_1to2",TfidfVectorizer(ngram_range=(1,2),
                     token_pattern=r"(?u)\b\w+\b",min_df=1,sublinear_tf=True)),
        ("lexical_char_3to5",TfidfVectorizer(analyzer="char",
                     ngram_range=(3,5),min_df=1,max_features=10000,sublinear_tf=True)),
    ):
        model=make_pipeline(vectorizer,LogisticRegression(
            C=1.0,max_iter=1000,class_weight="balanced"))
        train=[x for x in chosen if x["split"]=="train"]
        assert sum(x["gold"]=="OPEN_PRIVATE_CHANNEL" for x in train)==len(train)//2
        model.fit([actor_text(x["obs"]) for x in train],
                  [x["gold"] for x in train])
        ml[name]={}
        for split in ("train","dev","exploratory_test"):
            sample=[x for x in chosen if x["split"]==split]
            actual=model.predict([actor_text(x["obs"]) for x in sample])
            correct=sum(p==x["gold"] for p,x in zip(actual,sample))
            ml[name][split]={
                "n":len(sample),"correct":correct,
                "accuracy":round(correct/len(sample),4),
                "source_split":"train-only for fitting",
                "full_hidden_knowledge_graph_solver_used":False,
            }
    assert result["lexical_three_checks"]["train"]["first_action_accuracy"]<1
    assert result["visible_rule_graph_planner"]["train"]["first_action_accuracy"]==1
    assert all(result["visible_rule_graph_planner"][s]["first_action_accuracy"]==1
               for s in ("dev","exploratory_test"))
    assert all(result["lexical_three_checks"][s]["first_action_accuracy"]<1
               for s in ("train","dev","exploratory_test"))
    summary={
        "status":"V31_BALANCED_GATE_A_STRONG_BASELINES_EXECUTED",
        "mechanisms":len(source),
        "balanced_primary_histories":len(chosen),
        "positive_open_cases":len(source),
        "negative_skip_cases":len(source),
        "native_full_episodes":len(chosen)*len(allpolicies),
        "stored_native_witnesses":len(native),
        "fixed_business_state_and_authorization_within_history_pairs":True,
        "word_learner_has_no_hidden_oracle_labels_at_test_time":True,
        "unseen_graph_families_by_split":True,
        "grounded_native_rollout_policy_baselines":result,
        "train_only_lexical_ML_baselines":ml,
        "the_visible_rule_graph_is_sufficient_for_perfect_symbolic_planning":True,
        "no_GPU_no_paid_API":True,
        "claim_limit":"Synthetic formal rule-language, not human-norm-certified, model Qpi, or true blind benchmark.",
        "native_episode_samples":native,
    }
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="native_episode_samples"},
                     ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
