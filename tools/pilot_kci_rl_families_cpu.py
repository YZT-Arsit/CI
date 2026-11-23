r"""Read-only CPU diagnosis of candidate optimization families on a truly frozen 4B policy.
This compares available signals, never claims an algorithm's trained performance.
It uses archived native 4B trajectories and real executor transitions, not random toy rewards.
"""
from __future__ import annotations
import collections,hashlib,json,math,statistics,sys,tarfile
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import digest
from kci.agent_rl.state_restarts import RestartSpec,restore
BASE=ROOT/"artifacts"/"20261119_composition_signals"/"agent_rl_v1"/"policy_qpi_01"
OUT=BASE/"cpu_rl_family_pilot_20261119"
ARCHIVE=BASE/"state_directed_4b_probe_backup.tar.gz"
SEL=BASE/"selection_v2"/"FORKS.jsonl"
PREFIX="artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01/state_directed_4b_probe_01/"
def jlines(v):return [json.loads(s) for s in v.decode('utf-8').splitlines() if s.strip()]
def mean_or_none(x):return statistics.mean(x) if x else None
def run():
    assert ARCHIVE.exists() and SEL.exists()
    specs=[RestartSpec.from_record(x) for x in jlines(SEL.read_bytes())]
    assert len(specs)==16
    with tarfile.open(ARCHIVE,"r:gz") as t:
        def read(name):return t.extractfile(PREFIX+name).read()
        ep=jlines(read("episodes.jsonl"))
        groups=jlines(read("groups.jsonl"))
        manifest=json.loads(read("MANIFEST.json"))
        summary=json.loads(read("SUMMARY.json"))
        assert len(ep)==128 and len(groups)==16
        checked=0
        for p,h in manifest.items():
            assert hashlib.sha256(read(p)).hexdigest()==h,(p,"mismatch")
            checked+=1
        assert len({e["policy_id"] for e in ep})==1
    gate=collections.Counter()
    role=collections.defaultdict(collections.Counter)
    anchor_stats=[]
    dpo_pairs_total=0
    dpo_pairs_informative=0
    dpo_pairs_contradictory=0
    supports=collections.Counter()
    all_visits=[]
    immediate_events={}
    for gi,spec in enumerate(specs):
        g=groups[gi]
        assert g["group_index"]==gi and g["restart"]["state_hash"]==spec.state_hash
        eps=ep[gi*8:(gi+1)*8]
        assert len(eps)==8 and set(e["sample_index"] for e in eps)==set(range(8))
        assert all(e["initial_state_hash"]==spec.state_hash for e in eps)
        env,start=restore(spec)
        anchors=collections.defaultdict(list)
        for e in eps:
            _,state=restore(spec)
            for step,d in enumerate(e["decisions"]):
                assert digest(asdict(state))==d["state_hash"]
                assert env.actor(state)==d["actor"]
                assert tuple(d["actions"])==tuple(env.options(state))
                nxt,audit=env.transition(state,d["chosen_action"])
                assert audit["verdict"]==d["verdict"]
                assert json.loads(json.dumps(list(audit.get("new_protected_relations",[]))))==d["newly_derived"]
                v={
                    "group":gi,"sample":e["sample_index"],"step":step,
                    "role":d["actor"],"state_hash":d["state_hash"],
                    "action":d["chosen_action"],"reward":e["reward"],
                    "success":e["success"],"violation":e["violation"],
                    "branch":len(d["actions"])>1,
                    "new_derivation":bool(audit.get("new_protected_relations")),
                    "immediate_violation":audit["verdict"]=="VIOLATION",
                }
                all_visits.append(v)
                gate["all_decisions"]+=1
                gate["branch_decisions"]+=int(v["branch"])
                gate["new_derivation_steps"]+=int(v["new_derivation"])
                gate["violation_trigger_steps"]+=int(v["immediate_violation"])
                role[d["actor"]]["visits"]+=1
                role[d["actor"]]["violating_actions"]+=int(v["immediate_violation"])
                if v["branch"]:
                    role[d["actor"]]["branch_visits"]+=1
                if v["branch"]:
                    anchors[d["state_hash"]].append(v)
                state=nxt
            assert state.done
            assert bool(state.violation)==bool(e["violation"])
            assert bool(state.success)==bool(e["success"])
        assert sum(len(a) for a in anchors.values())==sum(
            len(d["actions"])>1 for e in eps for d in e["decisions"])
        for state_hash,visits in anchors.items():
            actions=sorted({v["action"] for v in visits})
            rewards=sorted({v["reward"] for v in visits})
            eligible=len(visits)>=2
            cross=len(actions)>=2
            informative=len(rewards)>=2
            supports["anchor_total"]+=1
            supports["anchor_repeated"]+=int(eligible)
            supports["anchor_multi_action"]+=int(cross)
            supports["anchor_reward_contrast"]+=int(informative)
            supports["anchor_multi_action_and_reward_contrast"]+=int(cross and informative)
            supports["branch_visits_in_multi_action_anchor"]+=len(visits) if cross else 0
            supports["branch_visits_in_signal_anchor"]+=len(visits) if cross and informative else 0
            localpos=localneg=0
            if cross:
                for i,a in enumerate(visits):
                    for b in visits[i+1:]:
                        if a["action"]==b["action"]:continue
                        d=a["reward"]-b["reward"]
                        dpo_pairs_total+=1
                        if abs(d)>=.5:
                            dpo_pairs_informative+=1
                        # fixed sorted action reference, so contradictory signs are meaningful
                        sign=(1 if d>.5 else -1 if d<-.5 else 0)
                        if a["action"]!=actions[0]:sign=-sign
                        localpos+=int(sign>0);localneg+=int(sign<0)
                dpo_pairs_contradictory+=int(localpos>0 and localneg>0)
            anchor_stats.append({
                "group":gi,"role":visits[0]["role"],
                "visits":len(visits),"multi_action":cross,"returns_vary":informative,
                "distinct_actions":actions,"mean_reward_by_action":{
                    a:statistics.mean(x["reward"] for x in visits if x["action"]==a) for a in actions},
                "comparisons_pref_action0":localpos,
                "comparisons_pref_action1":localneg,
                "naive_pairwise_preference_conflict":localpos>0 and localneg>0,
            })
        # Enumerate native one-step alternative transitions at every recorded on-policy
        # decision state exactly once, to see how many really have step-level
        # privacy distinctions; never use the result as a model action label.
        for e in eps:
            _,state=restore(spec)
            for d in e["decisions"]:
                key=(gi,d["state_hash"])
                if len(d["actions"])>1 and key not in immediate_events:
                    vals={}
                    for action in d["actions"]:
                        next_state,audit=env.transition(state,action)
                        assert audit["verdict"]!="UNKNOWN"
                        vals[action]={
                            "violates_now":audit["verdict"]=="VIOLATION",
                            "new_knowledge":bool(audit.get("new_protected_relations")),
                            "done":next_state.done,
                        }
                    immediate_events[key]={
                        "role":d["actor"],"related":spec.scenario.related,
                        "allow":spec.scenario.allow,
                        "actions":vals,
                        "has_unsafe_and_nonunsafe":len({v["violates_now"] for v in vals.values()})>=2,
                    }
                next_state,_=env.transition(state,d["chosen_action"])
                state=next_state
    event_multi=sum(int(v["has_unsafe_and_nonunsafe"]) for v in immediate_events.values())
    event_unsafe=sum(int(any(z["violates_now"] for z in v["actions"].values())) for v in immediate_events.values())
    assert gate["all_decisions"]==summary["decision_tokens"]
    assert gate["violation_trigger_steps"]==summary["violations"]
    metrics={
        "source":"Frozen Qwen3-4B-Instruct-2507-ms T=1 16 groups, 128 native episodes, 0 new model inference",
        "integrity":{"archive_sha256":hashlib.file_digest(open(ARCHIVE,"rb"),"sha256").hexdigest(),
                     "manifest_verified_files":checked,"native_transition_replayed_all":True,
                     "single_frozen_policy":True,"num_groups":len(groups),"num_episodes":len(ep)},
        "step_prm_signal":{
            "all_model_steps":gate["all_decisions"],
            "branch_steps":gate["branch_decisions"],
            "newly_derived_knowledge_steps":gate["new_derivation_steps"],
            "violation_trigger_steps":gate["violation_trigger_steps"],
            "unique_onpolicy_branch_states":len(immediate_events),
            "states_with_immediate_unsafe_alternative":event_unsafe,
            "states_with_safe_vs_unsafe_immediate_action":event_multi,
            "warning":"No knowledge increase is not automatically useful progress, and terminal violation already signals immediate unsafe transition in this executor."
        },
        "gigpo_anchor_signal":{
            **dict(supports),
            "step_support_fraction":supports["branch_visits_in_signal_anchor"]/gate["branch_decisions"],
            "strict_state_hash_and_same_scenario":True,
            "no_new_rollouts":True,
            "warning":"Grouping repeated exact states is already GiGPO's mechanism; future value differences are not necessarily due solely to the chosen action."
        },
        "step_dpo_offline_preferences":{
            "cross_action_same_state_pairs":dpo_pairs_total,
            "pairs_with_abs_terminal_reward_gap_ge_0_5":dpo_pairs_informative,
            "anchors_with_conflicting_pairwise_labels":dpo_pairs_contradictory,
            "warning":"Terminal outcome pairings within a shared state can disagree due to downstream sampled actions; robust counterfactual preference requires causal verification or repeated continuations."
        },
        "constrained_and_planning":{
            "all_sixteen_restart_states_have_safe_success_path":True, # checked below against exact
            "full_native_state_space_exhaustion_is_possible":True,
            "warning":"Search/planning with full hidden evaluator state is an optimistic privileged upper bound, not a deployable POMDP algorithm."
        },
        "role_step_signal":{k:dict(v) for k,v in role.items()},
        "anchor_examples":anchor_stats[:25],
    }
    # Independently establish nontrivial constrained feasibility from the exact native
    # solver at each start state, not from a copied frozen statement.
    for spec in specs:
        env,state=restore(spec)
        exact=env.exact(state)
        assert exact["status"]=="EXACT" and exact["witness"]
    assert metrics["constrained_and_planning"]["all_sixteen_restart_states_have_safe_success_path"]
    OUT.mkdir(parents=True,exist_ok=True)
    result=OUT/"FROZEN4B_RL_FAMILY_PILOT.json"
    if result.exists():raise FileExistsError(result)
    result.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k not in ("anchor_examples","role_step_signal")},
                     ensure_ascii=False,indent=2))
if __name__=="__main__":run()
