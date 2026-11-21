r"""Independent zero-model-training replay of 32 new public-task variants × 6 policies.

Load saved actual PyTorch model weights for 3 paired SFT and 3 CCE runs;
reconstruct each complete A->R->B->C signed business environment and
re-execute the actor's greedy native choices, including all recipients,
source checks, incremental CI and business outcomes.

Also redo all 4 exact native reference branches per stress world, to verify
regret. Deliberately does not choose/fix cases from test results.
"""
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import json,sys,torch,math
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.diverse_ci_branch_env import task_catalog,digest
from kci.diverse_ci_long_horizon_v31 import enumerate_specs_for_root
from kci.upstream_downstream_chain_pilot import ChainPilotEnv
from chain_shared_actor_cpu_action_interaction import SharedHornActor,run_episode
from solve_v31_longterm import native_best
DATA=ROOT/"artifacts"/"20261120_v31"/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"
P=DATA/"STRESS.json"
OUT=DATA/"INDEPENDENT_REPLAY_AUDIT.json"
if OUT.exists():raise FileExistsError(OUT)
j=json.loads(P.read_text(encoding="utf-8"))
assert j["status"]=="POSTHOC_REPAIRED_TRAINED_POLICY_VARIANT_STRESS_COMPLETE"
original=ROOT/"artifacts"/"20261120_v31"/"DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01"/"CHAIN_CAUSAL_PAIRS_EVAL_ONLY.jsonl"
sourcehash=sha256(original.read_bytes()).hexdigest()
tsk={t.id:t for t in task_catalog()}
ref={r["world"]:r for r in j["native_reference"]}
assert len(ref)==32
result={(r["method"],r["seed"],r["world"]):r for r in j["actual_policy_episodes"]}
assert len(result)==32*6
counts=Counter()
# Audit original source reference Q for NEW test variants, not only cached tags.
for world,saved in ref.items():
    case,mode,swap_text=world.split("__")
    root,vtext=case.split("_M")
    task=tsk[root];variant=int(vtext)-1
    assert variant in (3,4,5,6)
    assert task.graph in j["new_test_graph_families"]
    cond={"calendar":"early_open","retail":"pending_yes","document":"external"}[task.business]
    spec=enumerate_specs_for_root(task,cond,variant)
    env=ChainPilotEnv(spec,swap_text=="swap1",mode,4 if mode=="tradeoff" else 0)
    candidate=[]
    for package in ("PACKAGE_A","PACKAGE_B"):
        st=env.initial()
        for act in ("A_READ_VERIFIED_PACKAGES","A_SEND_"+package,
                    "INSPECT_BUSINESS","CHOOSE:"+env.native._route):
            st,a=env.transition(st,act)
            assert a["verdict"]=="VALID"
        for act in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"):
            r=native_best(env.native,st.native,act)
            candidate.append((r["native_exact_Qstar"],package,act,r["terminal"]["success"]))
    best=max(candidate)
    assert abs(best[0]-saved["best_reward"])<1e-7
    assert best[3] and saved["successful_native_witness"]
    counts["independently_recomputed_oracle_worlds"]+=1
for run in j["model_results"]:
    name=run["method"];seed=run["seed"]
    checkpoint=DATA/"SMALL_CPU_CHECKPOINTS"/(name+"_seed"+str(seed)+".pt")
    assert sha256(checkpoint.read_bytes()).hexdigest()==run["checkpoint_sha256"]
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert ck["source_data_sha256"]==sourcehash
    model=SharedHornActor()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    n=Counter();total_regret=0.
    for world in ref:
        case,mode,swap_text=world.split("__")
        root,vtext=case.split("_M")
        task=tsk[root];variant=int(vtext)-1
        spec=enumerate_specs_for_root(task,
             {"calendar":"early_open","retail":"pending_yes","document":"external"}[task.business],
             variant)
        env=ChainPilotEnv(spec,swap_text=="swap1",mode,4 if mode=="tradeoff" else 0)
        replay=run_episode(env,model,greedy=True)
        saved=result[(name,seed,world)]
        assert replay["actions"]==saved["actual_native_action_trace"]
        assert replay["success"]==saved["actual_native_success"]
        assert replay["violation"]==saved["actual_native_CI_violation"]
        assert replay["return"]==saved["actual_policy_return"]
        assert ref[world]["best_reward"]>=replay["return"]-1e-8
        n["episodes"]+=1
        n["safe_success"]+=int(replay["success"])
        n["ci_violations"]+=int(replay["violation"])
        n["incomplete"]+=int(not replay["success"] and not replay["violation"])
        total_regret+=ref[world]["best_reward"]-replay["return"]
        counts["trained_model_actual_native_trajectories_replayed"]+=1
        counts["trained_policy_actions_replayed"]+=len(replay["actions"])
    for key in ("episodes","safe_success","ci_violations","incomplete"):
        assert n[key]==run[key],(name,seed,key)
    assert abs(total_regret/32-run["mean_native_regret"])<1e-6
    counts["independent_saved_model_weights_verified"]+=1
report={
    "status":"INDEPENDENT_SAVED_PYTORCH_ACTOR_STRESS_NATIVE_REPLAY_PASS",
    "original_source_sha256":sourcehash,
    "counts":dict(counts),
    "new_variant_worlds":32,
    "source_graphs_still_two_developer_seen_families":True,
    "NOT_fresh_blind_test":True,
    "stored_neural_parameter_checkpoints_validated":True,
    "every_actor_native_action_reexecuted_from_synthetic_tool_source":True,
    "each_native_reference_Qstar_recomputed_not_trusted_metadata":True,
    "original_Qwen_weights_untouched":True,
}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,ensure_ascii=False),flush=True)
