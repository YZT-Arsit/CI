"""Read-only upstream A diagnostic. Check observation sufficiency, Q gaps, ties,
model actual package choices and neural feature behavior. No training.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,sys
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/"tools"),str(R/"code")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import actor_feature,environment
from kci.upstream_downstream_chain_pilot_public_contract import ChainPilotEnvPublicContract
BASE=R/"artifacts"/"20261120_v31"
RES=json.loads((BASE/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_ACTION_INTERACTION_01"/"RESULTS.json").read_text(encoding="utf-8"))
def choices(split,mode="action_sft",seed=7):
    rr=next(x for x in RES["per_seed_and_heldout_full_native_model_action_paths"] if x["seed"]==seed and x["mode"]==mode)
    return {x["world"]:next(y["action"].removeprefix("A_SEND_") for y in x["actual_model_actions"] if y["stage"]=="A_SEND" and y["learned"]) for x in rr["dev_test_episode_traces"][split]}
details=[];summ=defaultdict(Counter)
for split,worlds in GROUP.items():
    chosen={s:choices(split,"action_sft",s) for s in (7,1234,2025)} if split!="train" else {}
    for row in worlds:
        E=environment(row)
        st=E.initial();st,_=E.transition(st,"A_READ_VERIFIED_PACKAGES")
        obs=E.observation(st)
        contract=ChainPilotEnvPublicContract(E.spec,E.swap,E.source_mode,E.safe_surcharge).observation(st)
        q=row["upstream_A_values_given_exact_B_continuation"]
        gap=q["PACKAGE_A"]-q["PACKAGE_B"]
        feats={k:actor_feature(obs,"A_SEND_"+k) for k in ("PACKAGE_A","PACKAGE_B")}
        cfeats={k:actor_feature(contract,"A_SEND_"+k) for k in ("PACKAGE_A","PACKAGE_B")}
        r={
          "world":row["world"],"split":split,"graph":row["graph"],
          "business":row["business"],"mode":row["source_mode"],
          "swap":row["swapped_package_names"],
          "qA":q["PACKAGE_A"],"qB":q["PACKAGE_B"],
          "abs_gap":round(abs(gap),7),
          "preferred":"A" if gap>1e-8 else "B" if gap<-1e-8 else "tie",
          "risky_package":row["related_package"],
          "safe_package":row["benign_package"],
          "visible_source_package_costs":{k:obs["source_bundle_tool_output"][k]["cost_steps"] for k in ("PACKAGE_A","PACKAGE_B")},
          "public_channel_fee":E.spec.prep_fee,
          "public_step_cost":E.spec.cost_per_step,
          "A_raw_privacy_resource_contract_present":"privacy_resource_contract" in obs,
          "A_patched_privacy_resource_contract_present":"privacy_resource_contract" in contract,
          "source_feature_costs":{k:float(feats[k]["extra"][5]) for k in feats},
          "risk_or_safe_source":{k:("risky_for_downstream" if k==row["related_package"] else "safe_for_downstream") for k in feats},
          "contracted_feature_preflight_fee":{k:float(cfeats[k]["extra"][2]) for k in cfeats},
          "action_features_differ":bool(any(np.any(feats["PACKAGE_A"][k]!=feats["PACKAGE_B"][k]) for k in feats["PACKAGE_A"])),
          "policy_choices":{str(seed):chosen[seed].get(row["world"]) for seed in chosen},
        }
        for seed,choice_map in chosen.items():
            act=choice_map[row["world"]]
            summ[split,row["source_mode"],seed]["count"]+=1
            summ[split,row["source_mode"],seed]["correct"]+=int(abs(q[act]-max(q.values()))<1e-8)
            summ[split,row["source_mode"],seed]["picked_risky"]+=int(act==row["related_package"])
            summ[split,row["source_mode"],seed]["picked_A"]+=int(act=="PACKAGE_A")
        summ[split,row["source_mode"],"all"]["margin_nonzero"]+=int(abs(gap)>1e-8)
        summ[split,row["source_mode"],"all"]["tie"]+=int(abs(gap)<=1e-8)
        details.append(r)
O=BASE/"A_UPSTREAM_DECISION_LABEL_AND_OBSERVABILITY_AUDIT_01"
if O.exists():raise FileExistsError(O)
O.mkdir()
(O/"RAW_AUDIT.json").write_text(json.dumps({"rows":details,"group_summaries":{str(k):dict(v) for k,v in summ.items()}},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("SUMMARY",json.dumps({str(k):dict(v) for k,v in summ.items()},ensure_ascii=False,indent=2),flush=True)
print("TEST_ITEMS",json.dumps([x for x in details if x["split"]=="exploratory_test"],ensure_ascii=False,indent=2),flush=True)
