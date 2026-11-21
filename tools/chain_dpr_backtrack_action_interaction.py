r"""Conservative DPR from ACTUAL failed policy episodes, with BACKTRACK to B.

For a failed sampled A->R->B->C trajectory:
  - inspect actual chosen downstream source action and its real full state;
  - if no source-state legal correction can safely complete, backtrack to
    earlier B/Coordinator action state, not to the upstream legal A sender;
  - branch each relevant action using native verified finite reference search;
  - issue a same-state, same-role corrective preference ONLY when alternative
    strictly improves verified safe task return.

This is a CPU native-reference DPR *proxy*: not Qpi, not original shared-LLM DPR.
"""
from copy import deepcopy
from collections import Counter
import numpy as np
from chain_shared_actor_cpu_action_interaction import environment,run_episode,digest,score_actions
from compare_kci_chain_methods_cpu_action_interaction import point
from solve_v31_longterm import native_best

def collect(model,worlds,seed,limit=64):
    rng=np.random.default_rng(seed)
    records=[];count=Counter()
    for world in worlds:
        env=environment(world)
        episode=run_episode(env,model,rng,greedy=False)
        count["native_failed_policy_probe_episodes"]+=1
        count["actual_model_actions"]+=episode["n_policy_choices"]
        if episode["success"]:
            count["policy_successes"]+=1
            continue
        if episode["violation"]:
            count["actual_violations"]+=1
        else:
            count["safe_incomplete"]+=1
        # Backwards along actual execution, never replace legal A with an
        # imaginary oracle A just because the downstream model failed.
        for item in episode["decision_states"][::-1]:
            obs=item["actor_observation"]
            role=obs.get("actor",obs.get("role"))
            phase=obs.get("current_stage",obs.get("stage"))
            if phase not in ("release","preflight","choose_fact"):
                continue
            options=item["candidate_actions"]
            original=item["selected_action"]
            state=item["native_state_before"]
            if original not in options:continue
            curr,info=env.native.transition(state,original)
            if info["verdict"] not in ("VALID","VIOLATION"):continue
            candidate=[]
            # Native Q value conditional on the same actual state + role.
            oldval=native_best(env.native,state,original)
            count["DPR_reference_calls"]+=1
            count["DPR_reference_states_expanded"]+=oldval["states_explored"]
            for alt in options:
                if alt==original:continue
                other,a=env.native.transition(state,alt)
                if a["verdict"] not in ("VALID","VIOLATION"):continue
                corrected=native_best(env.native,state,alt)
                count["DPR_reference_calls"]+=1
                count["DPR_reference_states_expanded"]+=corrected["states_explored"]
                if corrected["native_exact_Qstar"]>oldval["native_exact_Qstar"]+1e-8 and (
                    corrected["terminal"]["success"] and not corrected["terminal"]["violation"]):
                    candidate.append((corrected["native_exact_Qstar"],alt,corrected))
            if not candidate:continue
            best=max(candidate,key=lambda x:x[0])
            newval,alternative,reference=best
            assert alternative in options and alternative!=original
            p=point(obs,alternative,
                    weight=newval-oldval["native_exact_Qstar"],
                    origin="DPR_ACTUAL_policy_fail_same_state_downstream_not_A",
                    rejected=original)
            assert p["actor"]==role
            assert phase in ("release","preflight","choose_fact")
            records.append(p)
            count["verified_DPR_same_state_pairs"]+=1
            count["DPR_localized_"+phase]+=1
            break
        else:
            count["unresolved_actual_failures"]+=1
        if len(records)>=limit:break
    unique={}
    for row in records:unique.setdefault((row["obs_sha256"],row["chosen"],row["rejected"]),row)
    count["unique_localized_prefs"]=len(unique)
    assert all(p["actor"]!="RecordsAgent" for p in unique.values())
    return list(unique.values()),dict(count)
