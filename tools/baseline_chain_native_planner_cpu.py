r"""Exact native reference baseline for all dynamic A->R->B->C test worlds.

A and B both decisions are chosen from ACTUAL verified native continuations,
not from a hard-coded task label. It is an offline exact oracle ceiling that
must not be represented as a deployable learned agent or a fair inference-cost
match to the small neural actors.
"""
from pathlib import Path
from collections import Counter,defaultdict
from dataclasses import asdict
from hashlib import sha256
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu import GROUP
from chain_shared_actor_cpu import environment,digest
from solve_v31_longterm import native_best
OUT=ROOT/"artifacts"/"20261120_v31"/"CHAIN_NATIVE_ORACLE_BASELINES_01"
if OUT.exists():raise FileExistsError(OUT)
rows=[];stats=defaultdict(Counter)
for split in ("train","dev","exploratory_test"):
    for world in GROUP[split]:
        env=environment(world)
        options=[]
        for pkg in ("PACKAGE_A","PACKAGE_B"):
            st=env.initial()
            st,report=env.transition(st,"A_READ_VERIFIED_PACKAGES")
            assert report["verdict"]=="VALID"
            st,report=env.transition(st,"A_SEND_"+pkg)
            assert report["verdict"]=="VALID"
            for a in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                st,report=env.transition(st,a)
                assert report["verdict"]=="VALID"
            for action in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"):
                val=native_best(env.native,st.native,action)
                options.append((val["native_exact_Qstar"],pkg,action,val))
        winning=max(options,key=lambda x:(x[0],-len(x[3]["action_trace"])))
        value,pkg,action,native=winning
        state=env.initial();steps=[]
        for a in ("A_READ_VERIFIED_PACKAGES","A_SEND_"+pkg,"INSPECT_BUSINESS",
                  "CHOOSE:"+env.native._route):
            n,r=env.transition(state,a)
            assert r["verdict"]=="VALID"
            steps.append({"actor":env.observation(state)["actor"],
                          "action":a,"native_before":digest(asdict(state.native)),
                          "native_after":digest(asdict(n.native))})
            state=n
        for part in native["action_trace"]:
            assert digest(asdict(state.native))==part["state_before_sha256"]
            n,r=env.transition(state,part["action"])
            assert r["verdict"]==part["actual_verdict"]
            steps.append({"actor":env.observation(state)["actor"],
                          "action":part["action"],
                          "native_before":digest(asdict(state.native)),
                          "native_after":digest(asdict(n.native))})
            state=n
        assert state.native.done
        assert abs(value-(-1.0 if state.native.violation else
                 1-env.spec.cost_per_step*state.native.steps if state.native.success else 0))<1e-6
        stats[split]["worlds"]+=1
        stats[split]["success"]+=int(state.native.success and not state.native.violation)
        stats[split]["violation"]+=int(state.native.violation)
        stats[split]["safe_incomplete"]+=int(not state.native.success and not state.native.violation)
        stats[split]["native_reference_branches_explored"]+=4
        stats[split]["native_states_search"]+=sum(x[3]["states_explored"] for x in options)
        rows.append({"world":world["world"],"split":split,
                     "optimal_A_package":pkg,"optimal_B_preflight":action,
                     "actual_native_Qstar":value,
                     "source_safety":"full original native receiving and database mutation",
                     "terminal_success":state.native.success,
                     "terminal_violation":state.native.violation,
                     "native_actions":steps})
result={
    "status":"EXACT_NATIVE_PILOT_REFERENCE_CEILING_VERIFIED",
    "split_counts":{k:dict(v) for k,v in stats.items()},
    "all_reference_choices_verified_by_actual_native_tool_transitions":True,
    "not_model_output_or_training":True,
    "not_compute_matched_to_shared_actor":True,
    "source_fixture_count":len(rows),
}
OUT.mkdir(parents=True)
(OUT/"TRACES_EVAL_ONLY.jsonl").write_text(
    "".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows),
    encoding="utf-8")
(OUT/"SUMMARY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,ensure_ascii=False),flush=True)
