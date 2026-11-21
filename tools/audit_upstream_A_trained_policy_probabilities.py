"""Read saved trained small CPU SFT/CCE policies; inspect upstream A probabilities.

All features read from model-visible signed A tool result; latent truth used
ONLY after scoring for value-gap/regret diagnostics. Also compare supplying
the actual public downstream setup cost at inference, without retraining.
This is descriptive, not a fresh blinded test.
"""
from pathlib import Path
from collections import defaultdict,Counter
from hashlib import sha256
import json,sys,torch,statistics
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/"code"),str(R/"tools")]
from chain_shared_actor_cpu_action_interaction import SharedHornActor,score_actions,environment
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci.upstream_downstream_chain_pilot_public_contract import ChainPilotEnvPublicContract
P=R/"artifacts"/"20261120_v31"/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/"SMALL_CPU_CHECKPOINTS"
OUT=R/"artifacts"/"20261120_v31"/"A_UPSTREAM_DECISION_LABEL_AND_OBSERVABILITY_AUDIT_01"
TARGET=OUT/"PROBABILITIES_TRAINED_SHARED_ACTOR.json"
if TARGET.exists():raise FileExistsError(TARGET)
records=[]
for method in ("action_sft","cce_reference_dpo_proxy"):
    for seed in (7,1234,2025):
        path=P/(method+"_seed"+str(seed)+".pt")
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        net=SharedHornActor()
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        for split,rows in GROUP.items():
            for world in rows:
                env=environment(world)
                initial=env.initial()
                state,resp=env.transition(initial,"A_READ_VERIFIED_PACKAGES")
                assert resp["verdict"]=="VALID"
                raw=env.observation(state)
                richer_env=ChainPilotEnvPublicContract(
                    env.spec,env.swap,env.source_mode,env.safe_surcharge)
                visible_contract=richer_env.observation(state)
                def scores(view):
                    with torch.no_grad():
                        logits=score_actions(net,view,["A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"])
                        return torch.softmax(logits,dim=0).tolist()
                pr=scores(raw);patched=scores(visible_contract)
                val=world["upstream_A_values_given_exact_B_continuation"]
                risky=world["related_package"]
                records.append({
                    "method":method,"seed":seed,"split":split,"world":world["world"],
                    "graph":world["graph"],"mode":world["source_mode"],
                    "swap":world["swapped_package_names"],"risky":risky,
                    "score_prob_PACKAGE_A":pr[0],"score_prob_PACKAGE_B":pr[1],
                    "score_prob_risky":pr[0] if risky=="PACKAGE_A" else pr[1],
                    "provided_public_contract_prob_PACKAGE_A":patched[0],
                    "raw_obs_had_public_privacy_setup_cost":False,
                    "estimated_action":("PACKAGE_A" if pr[0]>=pr[1] else "PACKAGE_B"),
                    "actual_observable_package_costs":{k:raw["source_bundle_tool_output"][k]["cost_steps"]
                         for k in ("PACKAGE_A","PACKAGE_B")},
                    "native_Qgap_eval_only":val["PACKAGE_A"]-val["PACKAGE_B"],
                    "direct_observation_sha256":__import__("kci.diverse_ci_branch_env",fromlist=["digest"]).digest(raw),
                })
s=defaultdict(list)
for rec in records:s[(rec["method"],rec["split"],rec["mode"])].append(rec)
summ={}
for key,items in s.items():
    summ[str(key)]={
        "n":len(items),"pred_risky_count":sum(x["estimated_action"]==x["risky"] for x in items),
        "mean_pred_risky_probability":round(statistics.mean(x["score_prob_risky"] for x in items),5),
        "mean_AB_probability_change_from_contract":round(statistics.mean(abs(x["score_prob_PACKAGE_A"]-x["provided_public_contract_prob_PACKAGE_A"]) for x in items),5),
    }
TARGET.write_text(json.dumps({
    "status":"SAVED_MODEL_A_ACTION_PROBABILITY_AUDIT",
    "saved_12_checkpoint_CI_data_training":"six actual small-model checkpoints (3 seeds x 2 methods)",
    "source_scores_legal_actor_only":True,
    "test_truth_only_for_eval":True,
    "records":records,
    "summary":summ,
    "no_model_retraining":True,
},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("A_PROB_SUMMARY",json.dumps(summ,ensure_ascii=False,indent=2),flush=True)
