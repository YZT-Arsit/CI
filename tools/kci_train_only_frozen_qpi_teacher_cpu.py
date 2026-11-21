r"""Freeze and calculate exact policy-conditional continuation credit on TRAIN ONLY.

A: same signed source-read state, force A_SEND_PACKAGE_A vs B; all subsequent
choices are actual SFT checkpoint π_ref probabilities (exact enumeration).
B: after each actual upstream package delivery, same Coordinator state,
force OPEN vs SKIP; remaining source/C actions use π_ref.

Qpi training labels constructed ONLY from these model-driven continuations.
Qstar is restricted to training/reference auditing, never used to choose Qpi
signs. Leave dev/exploratory tests untouched.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from collections import Counter
from hashlib import sha256
import json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_exact_frozen_qpi_cpu import (
 GROUP,SEEDS,environment,load_frozen_policy,ExactQpi,prepare_A,qstar_A_from_original)
from chain_shared_actor_cpu_action_interaction import digest
BASE=ROOT/"artifacts"/"20261121_policy_gap_pilot_01"
OUT=BASE/"TRAIN_ONLY_QPI_CREDIT_FROM_SFT_REFERENCE"
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir(parents=True)
assert [len(GROUP[s]) for s in ("train","dev","exploratory_test")]==[16,4,8]
assert {x["graph"] for x in GROUP["train"]}.isdisjoint(
    {x["graph"] for x in GROUP["dev"]+GROUP["exploratory_test"]})
audit=Counter()
for seed in SEEDS:
    model,meta=load_frozen_policy("action_sft",seed)
    frozen_before=sha256(json.dumps(
        {k:v.detach().cpu().tolist() for k,v in model.state_dict().items()},
        sort_keys=True).encode("utf-8")).hexdigest()
    records=[]
    begin=time.perf_counter()
    with (OUT/f"TRAIN_ONLY_SFT_REFERENCE_QPI_SEED{seed}.jsonl").open(
           "x",encoding="utf-8",buffering=1) as f:
        for row in GROUP["train"]:
            env=environment(row)
            source=prepare_A(env)
            Q=ExactQpi(env,model,meta["checkpoint_sha256"])
            A=Q.evaluate_A_arms(source)
            train_native_qstar=qstar_A_from_original(row,verify=True)
            records.append({
                "type":"A","world":row["world"],"graph":row["graph"],"split":"train",
                "role":"RecordsAgent","phase":"A_SEND",
                "source_state_hash":A["source_chain_hash"],
                "model_actor_observation":env.observation(source),
                "options":["A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"],
                "qpi":[A["arms"]["PACKAGE_A"]["Qpi"],A["arms"]["PACKAGE_B"]["Qpi"]],
                "qpi_gap":A["Qpi_A_minus_B"],
                "qstar_train_only_eval":[train_native_qstar["PACKAGE_A"],train_native_qstar["PACKAGE_B"]],
                "actual_R_after_forced_A":{
                    k:A["arms"][k]["actual_A_delivered_ledger"]
                    for k in ("PACKAGE_A","PACKAGE_B")},
                "comparison_source":"ONLY_FROZEN_SFT_MODEL_Qpi",
            })
            audit["A_same_state_train_pairs"]+=1
            for pkg in ("PACKAGE_A","PACKAGE_B"):
                st,rec=env.transition(source,"A_SEND_"+pkg)
                assert rec["verdict"]=="VALID"
                for a in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                    st,rec=env.transition(st,a)
                    assert rec["verdict"]=="VALID"
                obs=env.observation(st)
                assert obs["actor"]=="Coordinator" and st.native.stage=="preflight"
                assert {"OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"}<=set(obs["legal_actions"])
                values=[]
                qstar=[]
                for action in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"):
                    branch,rec=env.transition(st,action)
                    assert rec["verdict"]=="VALID"
                    values.append(Q.walk(branch)[0])
                    from solve_v31_longterm import native_best
                    qstar.append(native_best(env.native,st.native,action)["native_exact_Qstar"])
                records.append({
                    "type":"B","world":row["world"],"graph":row["graph"],
                    "split":"train","package":pkg,"role":"Coordinator","phase":"preflight",
                    "source_state_hash":digest(asdict(st)),
                    "model_actor_observation":obs,
                    "options":["OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"],
                    "qpi":values,"qpi_gap":round(values[0]-values[1],9),
                    "qstar_train_only_eval":qstar,
                    "actual_R_prior_from_A":st.native.ledger,
                    "comparison_source":"ONLY_FROZEN_SFT_MODEL_Qpi",
                })
                audit["B_same_state_train_pairs"]+=1
            audit["native_states_for_Qpi"]+=Q.count["native_states"]
            audit["native_actions_for_Qpi"]+=Q.count["actual_native_transition_edges"]
            audit["policy_softmax_calls"]+=Q.count["unique_frozen_actor_softmax_calls"]
            print("TEACHER",seed,row["world"],
                "A_gap",A["Qpi_A_minus_B"],"STATE_COST",Q.count["native_states"],flush=True)
        for r in records:
            assert r["split"]=="train"
            assert "Qpi" in r["comparison_source"]
            f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"),sort_keys=True)+"\n")
    frozen_after=sha256(json.dumps(
        {k:v.detach().cpu().tolist() for k,v in model.state_dict().items()},
        sort_keys=True).encode("utf-8")).hexdigest()
    assert frozen_before==frozen_after,"Reference weights changed during branch enumeration"
    assert len(records)==48
    nonzero_A=sum(r["type"]=="A" and abs(r["qpi_gap"])>1e-4 for r in records)
    nonzero_B=sum(r["type"]=="B" and abs(r["qpi_gap"])>1e-4 for r in records)
    summary={
       "seed":seed,"source_reference_checkpoint":meta,
       "train_worlds":16,"train_source_graphs":sorted({r["graph"] for r in GROUP["train"]}),
       "train_A_pairs":16,"train_B_pairs":32,
       "nontrivial_A_Qpi_gaps_1e-4":nonzero_A,
       "nontrivial_B_Qpi_gaps_1e-4":nonzero_B,
       "frozen_state_dict_checksum_before_after":frozen_before,
       "teacher_material_sha256":sha256(
            (OUT/f"TRAIN_ONLY_SFT_REFERENCE_QPI_SEED{seed}.jsonl").read_bytes()).hexdigest(),
       "elapsed_sec":round(time.perf_counter()-begin,2),
    }
    (OUT/f"TEACHER_RECEIPT_SEED{seed}.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SEED_TEACHER",json.dumps(summary,ensure_ascii=False),flush=True)
final={
 "status":"TRAIN_ONLY_EXACT_FROZEN_SFT_QPI_REFERENCE_CREDIT",
 "seeds":list(SEEDS),
 "total_train_A_pairs":audit["A_same_state_train_pairs"],
 "total_train_B_pairs":audit["B_same_state_train_pairs"],
 "total_Qpi_native_states":audit["native_states_for_Qpi"],
 "total_Qpi_native_actions":audit["native_actions_for_Qpi"],
 "total_frozen_actor_softmax_calls":audit["policy_softmax_calls"],
 "policy_id": "three saved SFT checkpoints, separately frozen per seed",
 "all_training_worlds_from_original_16":True,
 "test_Qstar_labels_used_for_training":False,
 "trainer_source_contains_no_dev_or_exploratory_records":True,
 "Qpi_values_are_not_native_Qstar":True,
}
(OUT/"SUMMARY.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("FINAL",json.dumps(final,ensure_ascii=False),flush=True)
