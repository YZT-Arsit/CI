"""A public-only planning baseline: no hidden Qstar labels, native search, or test tuning.

Reads ONLY RecordsAgent's verified source-package observation, upstream
disclosure costs, downstream required facts, public Horn rule clauses and
public privacy setup fee. Estimates cheapest safe suffix by projecting
mandatory source disclosures and paying for truthful private attestations
if the next raw source would complete PROTECTED. Compare to eval Qstar
only AFTER prediction. This detects whether 4/8 is a hard planning task.
"""
from pathlib import Path
from collections import Counter
import json,sys
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/"code"),str(R/"tools")]
from chain_shared_actor_cpu_action_interaction import environment
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci.upstream_downstream_chain_pilot_public_contract import ChainPilotEnvPublicContract

OUTPUT=R/"artifacts"/"20261120_v31"/"A_PUBLIC_ONLY_COST_PLANNER_P0_01"
if OUTPUT.exists():raise FileExistsError(OUTPUT)
def closure_visible(base,clauses):
    known=set(base)
    for _ in range(len(clauses)+2):
        fresh=set()
        for clause in clauses:
            if set(clause["requires"])<=known:
                fresh.add(clause["entails"])
        if fresh<=known:break
        known|=fresh
    return known

def predicted_variable_cost(observation,package):
    """Relative number of native steps without any Qstar or oracle access."""
    docs=observation["source_bundle_tool_output"]
    assert package in docs
    prior=set(docs[package]["fact_slots"])
    required=observation["downstream_required_business_checks"]
    rules=observation["public_inference_rules"]
    policy=observation["privacy_policy"]
    contract=observation["privacy_resource_contract"]
    authorized="Authorization exception is explicitly granted" in policy
    future_leak=("protected association" in closure_visible(prior|set(required),rules)
                 and not authorized)
    preflight=contract["private_channel_setup_steps"] if future_leak else contract["skip_setup_steps"]
    cost=docs[package]["cost_steps"]+preflight
    proven=[]
    raw_allowed_count=0
    verified_count=0
    for evidence in required:
        is_sensitive=("protected association" in closure_visible(prior|{evidence},rules)
                      and not authorized)
        if is_sensitive:
            assert future_leak
            cost+=3 # READ + VERIFY + SEND_CERT
            verified_count+=1
        else:
            prior.add(evidence)
            cost+=2 # READ + SEND_RAW
            raw_allowed_count+=1
    return {"relative_steps":cost,"future_leak":future_leak,
            "verified_count":verified_count,"raw_allowed_count":raw_allowed_count,
            "step_cost":contract["step_cost"]}

rows=[]
summ=Counter()
for split,examples in GROUP.items():
    for item in examples:
        old=environment(item)
        env=ChainPilotEnvPublicContract(
            old.spec,old.swap,old.source_mode,old.safe_surcharge)
        st=env.initial();st,resp=env.transition(st,"A_READ_VERIFIED_PACKAGES")
        assert resp["verdict"]=="VALID"
        obs=env.observation(st)
        assert "privacy_resource_contract" in obs
        plans={p:predicted_variable_cost(obs,p) for p in ("PACKAGE_A","PACKAGE_B")}
        predicted=min(plans,key=lambda p:(plans[p]["relative_steps"],p))
        true=item["upstream_A_values_given_exact_B_continuation"]
        oracle=max(true,key=lambda p:true[p])
        oracle_gap=abs(true["PACKAGE_A"]-true["PACKAGE_B"])
        predicted_regret=round(max(true.values())-true[predicted],7)
        computed_Q_gap=round(
           (plans["PACKAGE_B"]["relative_steps"]-plans["PACKAGE_A"]["relative_steps"])*plans["PACKAGE_A"]["step_cost"],7)
        true_Q_gap=round(true["PACKAGE_A"]-true["PACKAGE_B"],7)
        exact=(abs(computed_Q_gap-true_Q_gap)<1e-6)
        correct=predicted_regret<1e-6
        summ[split,"n"]+=1
        summ[split,"exact_value_gap"]+=int(exact)
        summ[split,"min_regret"]+=int(correct)
        summ[split,"strict_gap"]+=int(oracle_gap>1e-8)
        rows.append({"split":split,"world":item["world"],"graph":item["graph"],
                     "mode":item["source_mode"],"swap":item["swapped_package_names"],
                     "public_prediction":predicted,"planA":plans["PACKAGE_A"],
                     "planB":plans["PACKAGE_B"],
                     "predicted_cost_Qgap":computed_Q_gap,
                     "oracle_Qgap_EVAL_ONLY":true_Q_gap,
                     "exact_Qgap":exact,"correct":correct,"regret":predicted_regret,
                     "source_observation_sha256":__import__("kci.diverse_ci_branch_env",fromlist=["digest"]).digest(obs)})
        print("PUBLIC",item["world"],"PRED",predicted,
              "GAP",computed_Q_gap,true_Q_gap,
              "EXACT",exact,"REGRET",predicted_regret,flush=True)
result={
    "status":"ACTOR_VISIBLE_COST_PLANNER_SAME_KCI_P0_CASES",
    "source":"Only A public actor observation and certified source packets, never simulator hidden Q, future search or per-world label at inference",
    "counts":{str(k):v for k,v in summ.items()},
    "predictions":rows,
    "no_learning":True,"not_model_Qpi":True,
    "the_public_rules_are_explicit_and_have_exact_transparent_closure":True,
}
OUTPUT.mkdir(parents=True)
(OUTPUT/"RESULT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("SUMMARY",json.dumps(result["counts"],ensure_ascii=False,indent=2),flush=True)
