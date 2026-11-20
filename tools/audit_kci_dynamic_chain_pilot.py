"""Independent replay + actual same-state downstream correction for the A/B/C pilot.

All value labels use actual native transitions; compare a downstream RAW
breach and legally available private correction from identical receiver state.
A's earlier actual disclosure remains valid in both branches.
"""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.diverse_ci_branch_env import task_catalog,digest,FINAL,SLOT_LABELS
from kci.diverse_ci_long_horizon_v31 import enumerate_specs_for_root
from kci.upstream_downstream_chain_pilot import ChainPilotEnv
from solve_v31_longterm import native_best
BASE=ROOT/"artifacts"/"20261120_v31"/"DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01"
def load(n):
    return [json.loads(x) for x in (BASE/n).read_text(encoding="utf-8").splitlines() if x.strip()]
manifest=json.loads((BASE/"MANIFEST_SHA256.json").read_text(encoding="utf-8"))
for name,h in manifest.items():
    assert sha256((BASE/name).read_bytes()).hexdigest()==h
pairs=load("CHAIN_CAUSAL_PAIRS_EVAL_ONLY.jsonl")
source={x["world"]:x for x in load("A_SOURCE_DECISION_OBSERVATIONS.jsonl")}
obs={(x["world"],x["package"]):x for x in load("B_AFTER_ACTUAL_DELIVERY_OBSERVATIONS.jsonl")}
paths={(x["world"],x["package_selected"],x["downstream_B_preflight_action"]):x
       for x in load("REAL_AGENT_A_B_C_NATIVE_TRAJECTORIES_EVAL_ONLY.jsonl")}
assert (len(pairs),len(source),len(obs),len(paths))==(28,28,56,112)
TASK={t.id:t for t in task_catalog()}
num=Counter();correction=[]
for row in pairs:
    task=TASK[row["root"]]
    variant=int(row["world"].split("_M")[-1].split("__")[0])-1
    spec=enumerate_specs_for_root(task,{"calendar":"early_open","retail":"pending_yes","document":"external"}[task.business],variant)
    env=ChainPilotEnv(spec,row["swapped_package_names"],row["source_mode"],row["source_cost_surcharge"])
    first=env.initial()
    assert not first.native.ledger
    st,ar=env.transition(first,"A_READ_VERIFIED_PACKAGES")
    assert ar["verdict"]=="VALID"
    assert source[row["world"]]["current_model_input"]==env.observation(st)
    branches={}
    for pkg in ("PACKAGE_A","PACKAGE_B"):
        sel,a=env.transition(st,"A_SEND_"+pkg)
        assert a["verdict"]=="VALID" and not sel.native.violation
        assert all(m["sender"]=="RecordsAgent" and m["recipient"]=="R"
                   for m in sel.native.ledger)
        assert len(sel.source_delivered_ids)==len(sel.native.ledger)
        num["actual_legal_A_deliveries_rechecked"]+=1
        current=sel
        for act in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
            current,a=env.transition(current,act)
            assert a["verdict"]=="VALID"
        bo=env.observation(current)
        assert bo==obs[(row["world"],pkg)]["model_input"]
        assert bo["actor"]=="Coordinator"
        branch={}
        for action in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"):
            native=native_best(env.native,current.native,action)
            saved=paths[(row["world"],pkg,action)]
            assert saved["native_Qstar_after_A"]==native["native_exact_Qstar"]
            assert saved["native_full_actions_after_upstream"]==native["action_trace"]
            replay=current
            roles=set(["RecordsAgent"])
            for item in native["action_trace"]:
                roles.add(env.observation(replay)["actor"])
                assert digest(asdict(replay.native))==item["state_before_sha256"]
                replay,rep=env.transition(replay,item["action"])
                assert rep["verdict"]==item["actual_verdict"]
                assert digest(asdict(replay.native))==item["state_after_sha256"]
                num["native_action_transitions_replayed"]+=1
            assert replay.native.done and replay.native.business==saved["actual_final_business"]
            if replay.native.success:
                assert "Coordinator" in roles and "BusinessExecutor" in roles
                num["real_C_business_commits_replayed"]+=1
            branch[action]=native
            num["B_action_native_Qstar_arms_rechecked"]+=1
        risky=tuple(sel.source_bundles[pkg]["facts"])==spec.risky_prior
        gap=branch["OPEN_PRIVATE_CHANNEL"]["native_exact_Qstar"]-branch["SKIP_PRIVATE_CHANNEL"]["native_exact_Qstar"]
        assert (gap>0)==risky
        branches[pkg]={"risky":risky,"obs":bo,"gap":gap,"state":current}
        # For a risky upstream packet, imitate B's correctly authorized
        # preflight and consider its *downstream source-agent's* choice.
        if risky:
            downstream,a=env.transition(current,"OPEN_PRIVATE_CHANNEL")
            assert a["verdict"]=="VALID"
            for fact in spec.required:
                downstream,a=env.transition(downstream,"READ:"+SLOT_LABELS[fact])
                assert a["verdict"]=="VALID"
                raw,araw=env.transition(downstream,"SEND_RAW_R")
                if araw["verdict"]=="VIOLATION":
                    repaired,afix=env.transition(downstream,"VERIFY_PRIVATE_R")
                    assert afix["verdict"]=="VALID" and not repaired.native.violation
                    braw=native_best(env.native,downstream.native,"SEND_RAW_R")
                    bfix=native_best(env.native,downstream.native,"VERIFY_PRIVATE_R")
                    assert bfix["native_exact_Qstar"]>braw["native_exact_Qstar"]
                    correction.append({
                        "world":row["world"],"package":pkg,"source":fact,
                        "replay_before_action_sha256":digest(asdict(downstream.native)),
                        "A_upstream_was_valid":True,
                        "B_or_tool_agent_current_raw_is_actual_violation":True,
                        "corrected_agent_role":env.observation(downstream)["actor"],
                        "raw_action":"SEND_RAW_R","correction":"VERIFY_PRIVATE_R",
                        "raw_native_Qstar":braw["native_exact_Qstar"],
                        "corrected_native_Qstar":bfix["native_exact_Qstar"],
                        "original_same_state_and_receiver_history":True})
                    break
                assert araw["verdict"]=="VALID"
                downstream=raw
            else:raise AssertionError("Risky upstream branch has no downstream breach point")
    risk=next(x for x in branches.values() if x["risky"])
    safe=next(x for x in branches.values() if not x["risky"])
    assert risk["gap"]>0>safe["gap"]
    if row["source_mode"]=="matched":
        a=risk["obs"].copy();b=safe["obs"].copy()
        x=a.pop("delivered_message_log");y=b.pop("delivered_message_log")
        assert a==b and x!=y and len(x)==len(y)
        num["B_history_only_strict_flips"]+=1
    # Different packages are *both* legal in exactly one A source state;
    # no separate setup conditions, hidden chosen label, or oracle switch.
    num["same_A_source_state_two_legal_actions"]+=1
assert num["B_history_only_strict_flips"]==14
assert num["same_A_source_state_two_legal_actions"]==28
assert num["B_action_native_Qstar_arms_rechecked"]==112
assert len(correction)==28,len(correction)
out=BASE/"INDEPENDENT_CAUSAL_REPLAY_AUDIT.json"
if out.exists():raise FileExistsError(out)
with (BASE/"DOWNSTREAM_CORRECTION_PAIRS_EVAL_ONLY.jsonl").open("x",encoding="utf-8") as f:
    for x in correction:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
result={
    "status":"INDEPENDENT_DYNAMIC_CHAIN_CAUSAL_ACTION_REPLAY_PASS",
    "counts":dict(num),
    "actual_downstream_same_state_leak_vs_correction_pairs":len(correction),
    "real_upstream_A_delivery_is_legal":True,
    "downstream_B_value_changes_after_delivered_A_message":True,
    "downstream_agent_correction_targets_actual_violating_action":True,
    "business_executor_real_database_mutation_verified":True,
    "native_scripts_not_LLM_autonomous_episodes":True,
    "no_current_model_Qpi":True,
    "business_data_CI_norms_fictional":True,
}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("AUDIT",json.dumps(result,ensure_ascii=False),flush=True)
