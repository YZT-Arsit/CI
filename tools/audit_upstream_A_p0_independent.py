r"""Independent from-generator audit of upstream A P0 results and learning splits.

Recomputes all native Q gaps from the SOURCE public cost planner evidence,
verifies stored actor-only training logs against exact eval-only Q values,
checks same source root grouping, training/test graph disjointness, and
the tie-aware denominator that caused 4/8 to overstate decision competence.
"""
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import sys,json,re
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from chain_shared_actor_cpu_action_interaction import environment
from kci.upstream_downstream_chain_pilot_public_contract import ChainPilotEnvPublicContract
B=ROOT/"artifacts"/"20261120_v31"
SOURCE=B/"A_UPSTREAM_DECISION_LABEL_AND_OBSERVABILITY_AUDIT_01"
PUBLIC=B/"A_PUBLIC_ONLY_COST_PLANNER_P0_01"
TRAIN=B/"A_PUBLIC_COST_SFT_CCE_PAIRED_P0_01"
OUT=B/"A_UPSTREAM_P0_INDEPENDENT_VALIDATION_01"
if OUT.exists():raise FileExistsError(OUT)
actor=json.loads((SOURCE/"RAW_AUDIT.json").read_text(encoding="utf-8"))
planner=json.loads((PUBLIC/"RESULT.json").read_text(encoding="utf-8"))
learning=json.loads((TRAIN/"P0_A_SFT_VS_QSTAR.json").read_text(encoding="utf-8"))
graphfamilies={s:{w["graph"] for w in GROUP[s]} for s in GROUP}
assert all(not graphfamilies[a]&graphfamilies[b] for a,b in
           (("train","dev"),("train","exploratory_test"),("dev","exploratory_test")))
allrows={r["world"]:r for r in actor["rows"]}
planned={r["world"]:r for r in planner["predictions"]}
assert len(allrows)==len(planned)==28
stats=Counter()
for split,worlds in GROUP.items():
    for raw in worlds:
        world=raw["world"]
        r=allrows[world];p=planned[world]
        assert r["split"]==p["split"]==split
        assert r["abs_gap"]==round(abs(p["oracle_Qgap_EVAL_ONLY"]),7)
        assert p["exact_Qgap"] and p["correct"] and p["regret"]==0
        assert r["A_raw_privacy_resource_contract_present"] is False
        assert r["A_patched_privacy_resource_contract_present"] is True
        assert r["action_features_differ"]
        e=environment(raw)
        enriched=ChainPilotEnvPublicContract(e.spec,e.swap,e.source_mode,e.safe_surcharge)
        st=enriched.initial()
        st,report=enriched.transition(st,"A_READ_VERIFIED_PACKAGES")
        assert report["verdict"]=="VALID"
        obs=enriched.observation(st)
        assert r["public_channel_fee"]==obs["privacy_resource_contract"]["private_channel_setup_steps"]
        assert r["visible_source_package_costs"]=={
            k:obs["source_bundle_tool_output"][k]["cost_steps"]
            for k in ("PACKAGE_A","PACKAGE_B")}
        stats["same_original_source_worlds"]+=1
        stats[split+"_worlds"]+=1
        stats[split+"_strict"]+=int(abs(r["qA"]-r["qB"])>1e-8)
        stats[split+"_ties"]+=int(abs(r["qA"]-r["qB"])<=1e-8)
        if split=="exploratory_test":
            assert set(r["policy_choices"])=={"7","1234","2025"}
            for seed,choice in r["policy_choices"].items():
                stats["original_shared_actor_picked_risky"]+=int(choice==r["risky_package"])
                stats["original_shared_actor_strict_optimal"]+=int(abs(
                    raw["upstream_A_values_given_exact_B_continuation"][choice]-
                    max(raw["upstream_A_values_given_exact_B_continuation"].values()))<1e-8)
        stats["public_planner_numerical_gap_exact"]+=int(abs(
            p["predicted_cost_Qgap"]-p["oracle_Qgap_EVAL_ONLY"])<1e-7)
assert stats["public_planner_numerical_gap_exact"]==28
assert stats["exploratory_test_strict"]==6 and stats["exploratory_test_ties"]==2
assert stats["original_shared_actor_picked_risky"]==24
assert stats["original_shared_actor_strict_optimal"]==12 # includes six tie points
assert learning["train_strict_action_states"]==14
assert learning["test_strict_states"]==6
for label,val in learning["aggregate"].items():
    assert len(val["test_strict_correct_per_seed"])==5
    assert all(0<=x<=6 for x in val["test_strict_correct_per_seed"])
    assert all(0<=x<=8 for x in val["test_all_correct_per_seed"])
    for seed_data in (r for r in learning["all_seed_checkpoints"]
                      if str(r["full_visible_cert_count"])+"__"+r["mode"]==label):
        final=seed_data["checkpoints"][-1]["exploratory_test"]
        assert final["strict_correct"]==val["test_strict_correct_per_seed"][
            [7,1234,2025,42,999].index(seed_data["seed"])]
        regrets=[]
        for predicted in final["chosen_probabilities_and_eval_Qgaps"]:
            row=allrows[predicted["world"]]
            pA=predicted["p_PACKAGE_A"]
            chosen="PACKAGE_A" if pA>=0.5 else "PACKAGE_B"
            q=row["qA"] if chosen=="PACKAGE_A" else row["qB"]
            regret=round(max(row["qA"],row["qB"])-q,8)
            assert regret==predicted["regret"]
            regrets.append(regret)
        assert abs(sum(regrets)/8-final["mean_regret"])<1e-7
        stats["learned_model_checkpoints_rechecked"]+=1
assert stats["learned_model_checkpoints_rechecked"]==40
assert learning["aggregate"]["True__optimal_action_SFT"]["test_strict_correct_per_seed"]==[6]*5
assert learning["aggregate"]["True__Qgap_weighted_SFT"]["test_strict_correct_per_seed"]==[4]*5
assert learning["aggregate"]["True__Qgap_reference_pair_DPO_proxy"]["test_strict_correct_per_seed"]==[6]*5
assert learning["aggregate"]["False__optimal_action_SFT"]["test_strict_correct_per_seed"]==[6]*5
OUT.mkdir()
receipt={
    "status":"INDEPENDENT_UPSTREAM_A_P0_SOURCE_AND_RESULT_AUDIT_PASS",
    "source_datasets_frozen":True,
    "no_4B_model_training_or_new_GPU":True,
    "no_blind_test":True,
    "public_cost_planner_no_future_native_search":True,
    "graph_family_split_disjoint":True,
    "counts":dict(stats),
    "raw_actor_probability_source":SOURCE.name+"/PROBABILITIES_TRAINED_SHARED_ACTOR.json",
    "source_sha256":{
        str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest()
        for path in (
            SOURCE/"RAW_AUDIT.json",
            SOURCE/"PROBABILITIES_TRAINED_SHARED_ACTOR.json",
            PUBLIC/"RESULT.json",
            TRAIN/"P0_A_SFT_VS_QSTAR.json",
        )},
}
(OUT/"VALIDATION.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in receipt.items() if k!="source_sha256"},ensure_ascii=False,indent=2),flush=True)
