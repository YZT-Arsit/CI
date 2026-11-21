r"""INDEPENDENT exhaustive native-path replay of frozen + trained CPU policy Qpi.

Does not call ExactQpi.walk or reuse its memo table. A separate DFS expands
actual tool/agent actions and multiplies their current checkpoint probabilities.
Unlike a sampled test run, every reachable terminal path contributes its own
native return, violation, success and business result. Verifies saved hashes,
roles, receipt prefixes and task success. Qstar is used ONLY for offline
native regret/conflict checks.

Audit scope: all 28×6 ORIGINAL reference checkpoints, and the 8 developer-
inspected exploratory worlds under 4 methods × 3 trained checkpoints. Each
policy is explicitly loaded from saved source-checkpoint SHA256.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from collections import Counter,defaultdict
from hashlib import sha256
import json,sys,math,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from chain_shared_actor_cpu_action_interaction import (
    SharedHornActor,environment,score_actions,
    valid_decision_candidates,baseline_automatic_action,digest)
from kci_exact_frozen_qpi_cpu import (
  load_frozen_policy,prepare_A,ROWS,GROUP,SEEDS,KINDS)
BASE=ROOT/"artifacts"/"20261121_policy_gap_pilot_01"
SAVED=BASE/"SIX_SAVED_CHECKPOINTS_QSTAR_VS_EXACT_QPI"/"ALL_SIX_FROZEN_QPI_EVIDENCE.jsonl"
POST=BASE/"EXACT_QPI_BEFORE_AFTER_FOUR_METHODS_02"/"ALL_EXACT_PRE_POST_POLICY_REEVALUATIONS.jsonl"
TRAIN=BASE/"FOUR_WAY_CPU_TRAINED_POLICY_RESULTS_03"
OUT=BASE/"INDEPENDENT_EXHAUSTIVE_QPI_NATIVE_REPLAY"
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir(parents=True)
def jlines(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
before=jlines(SAVED)
assert len(before)==28*6
after=[x for x in jlines(POST) if x["policy_state"]=="after" and x["split"]=="exploratory_test"]
assert len(after)==4*3*8

def walk_direct(env,state,model,stats,depth=0):
    if depth>=64:raise RuntimeError("Exceeded valid native episode horizon")
    st=state.native
    if st.done:
        assert all(log["fact"] in st.delivered[log["recipient"]] for log in st.ledger)
        if st.success:assert st.generated_result and not st.violation
        value=float(round(1-env.spec.cost_per_step*st.steps,6)) if st.success else (
            -1.0 if st.violation else 0.0)
        stats["actual_terminal_leaves_checked"]+=1
        return (value,float(st.success and not st.violation),float(st.violation),
                float(not st.success and not st.violation))
    visible=env.observation(state)
    role=visible.get("actor",visible.get("role"))
    assert role
    actions=list(env.options(state))
    selections=valid_decision_candidates(visible,actions)
    if len(selections)>=2:
        with torch.no_grad():
            logits=score_actions(model,visible,selections)
            assert torch.isfinite(logits).all()
            ps=torch.softmax(logits.to(torch.float64),dim=0).tolist()
        assert abs(sum(ps)-1)<1e-12
        options=selections
        stats["independent_policy_distribution_uses"]+=1
    else:
        a=baseline_automatic_action(env,state)
        assert a in actions
        options=[a];ps=[1.]
    output=[0.,0.,0.,0.]
    original_state_hash=digest(asdict(state))
    original_ledger=list(st.ledger)
    for a,p in zip(options,ps):
        nextstate,receipt=env.transition(state,a)
        assert receipt["verdict"] in ("VALID","VIOLATION"),receipt
        assert digest(asdict(state))==original_state_hash
        assert state.native.ledger==original_ledger
        assert nextstate.native.ledger[:len(original_ledger)]==original_ledger
        assert nextstate.native.steps>st.steps or nextstate.native.done
        for actual in nextstate.native.ledger[len(original_ledger):]:
            assert actual["recipient"] in ("R","M")
            assert actual["fact"] in nextstate.native.delivered[actual["recipient"]]
        stats["native_edges_reexecuted"]+=1
        scores=walk_direct(env,nextstate,model,stats,depth+1)
        for i,v in enumerate(scores):output[i]+=p*v
    assert abs(sum(output[1:])-1)<1e-7
    return tuple(output)

models={}
def original_model(kind,seed):
    key=(kind,seed)
    if key not in models:
        models[key]=load_frozen_policy(kind,seed)
    return models[key]

for rec in before:
    method,seed=rec["method"],rec["seed"]
    model,meta=original_model(method,seed)
    assert rec["checkpoint"]["checkpoint_sha256"]==meta["checkpoint_sha256"]
    env=environment(ROWS[rec["world"]])
    source=prepare_A(env)
    assert rec["frozen_exact_Qpi"]["source_chain_hash"]==digest(asdict(source))
    sums={}
    stats=Counter()
    for pkg in ("PACKAGE_A","PACKAGE_B"):
        child,receipt=env.transition(source,"A_SEND_"+pkg)
        assert receipt["verdict"]=="VALID"
        sums[pkg]=walk_direct(env,child,model,stats)
        saved=rec["frozen_exact_Qpi"]["arms"][pkg]
        assert abs(sums[pkg][0]-saved["Qpi"])<1e-6,(rec["world"],method,seed,pkg,sums[pkg],saved["Qpi"])
        assert abs(sums[pkg][1]-saved["success_probability"])<1e-6
        assert abs(sums[pkg][2]-saved["CI_violation_probability"])<1e-6
    stats["original_worlds_verified"]+=1
    if not models.get(("stats",method,seed)):models[("stats",method,seed)]=Counter()
    models[("stats",method,seed)].update(stats)
print("REPLAY_ALL_SIX_FROZEN",sum(models.get(("stats",k,s),Counter())["original_worlds_verified"]
      for k in KINDS for s in SEEDS),flush=True)

postmodels={}
poststats=Counter()
for rec in after:
    key=(rec["policy_kind"],rec["seed"])
    if key not in postmodels:
        path=TRAIN/"MODEL_WEIGHTS"/f"{key[0]}_seed{key[1]}.pt"
        ckpt=torch.load(path,map_location="cpu",weights_only=True)
        assert ckpt["seed"]==key[1] and ckpt["method"]==key[0]
        model=SharedHornActor()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        postmodels[key]=(model,sha256(path.read_bytes()).hexdigest())
    model,fingerprint=postmodels[key]
    assert fingerprint==rec["checkpoint_sha256"]
    env=environment(ROWS[rec["world"]])
    source=prepare_A(env)
    stats=Counter()
    for pkg in ("PACKAGE_A","PACKAGE_B"):
        child,a=env.transition(source,"A_SEND_"+pkg)
        assert a["verdict"]=="VALID"
        res=walk_direct(env,child,model,stats)
        saved=rec["forced_A_native_Qpi_and_full_witness"]["arms"][pkg]
        assert abs(res[0]-saved["Qpi"])<1e-6,(key,rec["world"],pkg,res,saved["Qpi"])
        assert abs(res[1]-saved["success_probability"])<1e-6
        assert abs(res[2]-saved["CI_violation_probability"])<1e-6
    full=walk_direct(env,env.initial(),model,stats)
    assert abs(full[0]-rec["full_current_policy_native_Qpi"])<1e-6
    assert abs(full[1]-rec["full_current_policy_success_probability"])<1e-6
    assert abs(full[2]-rec["full_current_policy_leak_probability"])<1e-6
    poststats["post_test_world_checkpoints_verified"]+=1
    poststats["actual_native_branches_replayed"]+=stats["native_edges_reexecuted"]
    poststats["terminal_business_leaves_checked"]+=stats["actual_terminal_leaves_checked"]
assert poststats["post_test_world_checkpoints_verified"]==12*8
summary={
  "status":"INDEPENDENT_FROZEN_AND_TRAINED_Qpi_NATIVE_DFS_REPLAY_PASS",
  "before_original_checkpoints":[k+"_"+str(s) for k in KINDS for s in SEEDS],
  "before_worlds_verified":168,
  "before_by_checkpoint":{k+"_"+str(s):dict(models[("stats",k,s)]) for k in KINDS for s in SEEDS},
  "after_new_model_checkpoints":len(postmodels),
  "after_exploratory_worlds_verified":poststats["post_test_world_checkpoints_verified"],
  "after_stats":dict(poststats),
  "independent_DFS_not_original_ExactQpi_DP":True,
  "native_source_receipts_and_role_action_verdicts_reexecuted":True,
  "all_expected_terminal_rewards_recomputed":True,
  "no_future_optimal_action_used_by_frozen_policy":True,
  "data_and_benchmark_unchanged":True,
  "does_not_test_Qwen_LLM_Qpi":True,
}
(OUT/"SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("AUDIT",json.dumps({k:v for k,v in summary.items() if k!="before_by_checkpoint"},
    ensure_ascii=False),flush=True)
