r"""Read-only CPU analysis of archived KCI 16-group state-directed GRPO vs PCCE.
No GPU, checkpoints, API or training. Fails closed on integrity/identity mismatches.
Usage: py -3 tools\diagnose_state_matched_cpu.py
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import random
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "20261119_composition_signals" / "agent_rl_v1" / "policy_qpi_01"
ARCHIVE = BASE / "state_matched16_audit_backup_20261119.tar.gz"
OUT = BASE / "cpu_state_matched16_diagnostic_01"
ARMS = ("terminal", "policy_cce")
PREFIX = "artifacts/20261119_composition_signals/agent_rl_v1/"
EXPECTED_SHA = "a5e6e908d0bdaecd6e3f3c6af84450e75a88c1af1bc5319ad1d45a274b000543"

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def safe_load(tf, path):
    member = tf.getmember(path)
    if not member.isfile():
        raise ValueError("Non-file archive entry: " + path)
    with tf.extractfile(member) as f:
        return f.read()

def jread(tf, path):
    return json.loads(safe_load(tf, path))

def jlines(tf, path):
    return [json.loads(s) for s in safe_load(tf, path).decode("utf-8-sig").splitlines() if s.strip()]

def armfile(arm, rel):
    return PREFIX + "state_restart_" + arm + "_16_01/" + rel

def sgn(v, tol=1e-9):
    return (v > tol) - (v < -tol)

def outcome(ep):
    return "violation" if ep["violation"] else "safe_success" if ep["success"] else "safe_incomplete"

def identity(ep):
    return ep["scenario_key"], ep["sample_index"]

def path_signature(ep):
    return [(d["state_hash"], d["chosen_action"]) for d in ep["decisions"]]

def index_dev(episodes):
    out = {identity(e): e for e in episodes}
    assert len(out) == len(episodes), "Duplicate DEV scenario/sample"
    return out

def load(tf, arm):
    return dict(
        summary=jread(tf, armfile(arm, "SUMMARY.json")),
        config=jread(tf, armfile(arm, "RUN_CONFIG.json")),
        groups=jlines(tf, armfile(arm, "groups.jsonl")),
        train=jlines(tf, armfile(arm, "episodes.jsonl")),
        before=index_dev(jlines(tf, armfile(arm, "dev_before/episodes.jsonl"))),
        after=index_dev(jlines(tf, armfile(arm, "dev_after/episodes.jsonl"))),
    )

def match_groups(data):
    pos = 0
    output = {}
    for g in data["groups"]:
        eps = data["train"][pos : pos + g["sample_size"]]
        assert len(eps) == g["sample_size"] == 8
        assert {e["sample_index"] for e in eps} == set(range(8))
        assert all(e["scenario_key"] == g["group_id"] and e["initial_state_hash"] == g["source_state_hash"]
                   and e["policy_id"] == g["policy_id"] for e in eps)
        output[g["group_index"]] = sorted(eps, key=lambda e: e["sample_index"])
        pos += len(eps)
    assert pos == len(data["train"]) == 128 and len(output) == 16
    return output

def matched_compare(left, right):
    assert set(left) == set(right)
    changed_first, changed_path, changed_outcome, changed_reward = [], [], [], []
    a_only_success, b_only_success, a_only_viol, b_only_viol = [], [], [], []
    first_diff_role = Counter()
    for key in sorted(left):
        a, b = left[key], right[key]
        assert a["rollout_seed"] == b["rollout_seed"]
        assert a["initial_state_hash"] == b["initial_state_hash"]
        if a["decisions"][0]["chosen_action"] != b["decisions"][0]["chosen_action"]:
            changed_first.append(key)
        if path_signature(a) != path_signature(b):
            changed_path.append(key)
            for da, db in zip(a["decisions"], b["decisions"]):
                if da["state_hash"] != db["state_hash"]:
                    break
                if da["chosen_action"] != db["chosen_action"]:
                    first_diff_role[da["actor"]] += 1
                    break
        if outcome(a) != outcome(b):
            changed_outcome.append((key, outcome(a), outcome(b)))
        if a["reward"] != b["reward"]:
            changed_reward.append(key)
        if a["success"] and not a["violation"] and not (b["success"] and not b["violation"]):
            a_only_success.append(key)
        if b["success"] and not b["violation"] and not (a["success"] and not a["violation"]):
            b_only_success.append(key)
        if a["violation"] and not b["violation"]:
            a_only_viol.append(key)
        if b["violation"] and not a["violation"]:
            b_only_viol.append(key)
    return {
        "episodes":len(left),
        "different_first_action":len(changed_first),
        "different_full_native_action_path":len(changed_path),
        "different_terminal_outcome":len(changed_outcome),
        "different_terminal_reward":len(changed_reward),
        "A_only_safe_success":len(a_only_success),
        "B_only_safe_success":len(b_only_success),
        "A_only_CI_violation":len(a_only_viol),
        "B_only_CI_violation":len(b_only_viol),
        "first_diverging_role":dict(first_diff_role),
        "outcome_diff_examples":[
            {"scenario":key[0],"sample_index":key[1],"A":oa,"B":ob}
            for key,oa,ob in changed_outcome[:30]],
    }

def bootstrap_scenario_clusters(a, b, n_boot=4000):
    keys=sorted(a)
    scenarios=sorted({k[0] for k in keys})
    per={}
    for s in scenarios:
        xs=[k for k in keys if k[0]==s]
        per[s]={
            "success_delta":sum(int(b[k]["success"] and not b[k]["violation"])
                                -int(a[k]["success"] and not a[k]["violation"]) for k in xs) / len(xs),
            "violation_delta":sum(int(b[k]["violation"])-int(a[k]["violation"]) for k in xs)/len(xs),
            "n":len(xs),
        }
    assert all(row["n"]==8 for row in per.values())
    rng=random.Random(9111926)
    sampling={"success_delta":[],"violation_delta":[]}
    for _ in range(n_boot):
        selected=[rng.choice(scenarios) for _ in scenarios]
        for metric in sampling:
            sampling[metric].append(sum(per[s][metric] for s in selected)/len(selected))
    interval={}
    for metric,values in sampling.items():
        values.sort()
        interval[metric]={
            "observed":sum(per[s][metric] for s in scenarios)/len(scenarios),
            "scenario_cluster_bootstrap_95pct":[values[int(.025*n_boot)],values[int(.975*n_boot)]],
        }
    return {"n_scenarios":len(scenarios),"n_trajectories":len(keys),
            "descriptive_only_not_independent_task_family":True,
            "B_minus_A":interval}

def main():
    assert ARCHIVE.is_file()
    archive_hash=hashlib.file_digest(ARCHIVE.open("rb"),"sha256").hexdigest()
    assert archive_hash==EXPECTED_SHA, "Backup checksum changed"
    with tarfile.open(ARCHIVE,"r:gz") as tf:
        names=set(tf.getnames())
        integrity={}
        for arm in ARMS:
            manifest=jread(tf,armfile(arm,"MANIFEST.json"))
            checks={}
            for rel,digest in manifest.items():
                actual=sha256(safe_load(tf,armfile(arm,rel)))
                if actual!=digest:
                    raise AssertionError("Inner manifest mismatch: "+arm+"/"+rel)
                checks[rel]=actual
            integrity[arm]={"verified_files":len(checks),"all_manifest_sha256_pass":True}
        arm={kind:load(tf,kind) for kind in ARMS}
        proofs=jlines(tf,armfile("policy_cce","restart_cce_proofs.jsonl"))
        matched=jread(tf,PREFIX+"STATE_DIRECTED_MATCHED16_REPORT.json")
    A,B=arm["terminal"],arm["policy_cce"]
    assert matched["matched_initial_DEV"] is True
    assert len(proofs)==B["summary"]["accepted_local_credit_decisions"]==64
    assert all(A["config"][k]==B["config"][k] for k in
               ["seed","max_groups","group_size","dev_world_limit","temperature",
                "explore_epsilon","lr","lora_rank","lora_alpha","selection_sha256","source_hashes"])
    assert A["before"].keys()==B["before"].keys()==A["after"].keys()==B["after"].keys()
    assert all(path_signature(A["before"][k])==path_signature(B["before"][k])
               and outcome(A["before"][k])==outcome(B["before"][k])
               for k in A["before"]), "DEV base trajectories not identical"
    groups_A=match_groups(A)
    groups_B=match_groups(B)
    for i in range(16):
        ag,bg=A["groups"][i],B["groups"][i]
        assert ag["group_index"]==bg["group_index"]==i
        assert ag["restart"]==bg["restart"]
        assert [e["rollout_seed"] for e in groups_A[i]]==[e["rollout_seed"] for e in groups_B[i]]
    credits={(p["group_index"],p["sample_index"]):p for p in proofs}
    assert len(credits)==len(proofs)
    roles=defaultdict(lambda:Counter())
    sign_counts=Counter()
    supported=defaultdict(list)
    records=[]
    proxy_terminal_abs=0.0
    proxy_local_abs=0.0
    first_terminal_abs=0.0
    first_local_abs=0.0
    all_branch_actions=0
    train_match={}
    for i in range(16):
        ga,gb=A["groups"][i],B["groups"][i]
        eps_A,eps_B=groups_A[i],groups_B[i]
        train_match[i]=matched_compare(
            {identity(e):e for e in eps_A},
            {identity(e):e for e in eps_B})
        rr=[e["reward"] for e in eps_B]
        mu=statistics.mean(rr); sigma=statistics.pstdev(rr)
        assert sigma>1e-10, "Unexpected zero-variance group"
        for ep in eps_B:
            sid=ep["sample_index"]
            adv=(ep["reward"]-mu)/(sigma+1e-8)
            branch=sum(len(d["actions"])>1 for d in ep["decisions"])
            if branch:
                w=1/(8*branch)
                proxy_terminal_abs+=abs(adv)*w*branch
                all_branch_actions+=branch
                if len(ep["decisions"][0]["actions"])>1:
                    first_terminal_abs+=abs(adv)*w
            proof=credits.get((i,sid))
            if not proof:continue
            assert proof["decision_index"]==0 and proof["restart_state_hash"]==ep["initial_state_hash"]
            assert proof["chosen_action"]==ep["decisions"][0]["chosen_action"]
            assert proof["native_Qstar_used"] is False
            assert proof["policy_id"]==ep["policy_id"]
            credit=proof["local_credit"]
            additive=0.5*credit
            net=adv+additive
            if branch:
                proxy_local_abs+=abs(additive)*w
                first_local_abs+=abs(additive)*w
            sign_counts["credited"]+=1
            if sgn(adv)==sgn(credit):k="same_sign"
            elif sgn(adv)==0:k="terminal_approximately_zero"
            else:k="opposite_sign"
            sign_counts[k]+=1
            if sgn(adv)*sgn(net)<0:sign_counts["net_advantage_sign_reversed"]+=1
            if sgn(adv)*sgn(credit)<0 and abs(net)<abs(adv):
                sign_counts["terminal_direction_attenuated"]+=1
            if sgn(adv)*sgn(credit)>0:
                sign_counts["terminal_direction_amplified"]+=1
            roles[proof["role"]][k]+=1
            roles[proof["role"]]["credited"]+=1
            supported[i].append(credit)
            evidence=proof["Qpi_evidence"]
            if "signal_gate" in evidence:
                assert evidence["signal_gate"]["supported"]
            records.append({
                "group":i,"sample":sid,"pair_id":gb["restart"]["pair_id"],
                "role":proof["role"],"history_related":gb["restart"]["scenario"]["related"],
                "scenario":ep["scenario_key"],"action":proof["chosen_action"],
                "terminal_reward":ep["reward"],"terminal_standardized_advantage":adv,
                "policy_local_raw_credit":credit,"weighted_local_addition":additive,
                "combined_first_action_coefficient":net,"relation_to_terminal":k,
                "net_sign_reversed":sgn(adv)*sgn(net)<0,
                "per_decision_loss_weight":1/(8*branch) if branch else 0,
            })
    assert len(supported)==B["summary"]["supported_counterfactual_initial_states"]==8
    assert len(supported)==8 and all(len(v)==8 for v in supported.values())
    assert all(B["groups"][i]["state_cce_evidence"]["supported_prefixes"]==int(i in supported)
               for i in range(16))
    assert all(B["groups"][i]["cce_nonzero_local_decisions"]==len(supported.get(i,[]))
               for i in range(16))
    gate_by_role=defaultdict(lambda:Counter())
    gate_by_history=defaultdict(lambda:Counter())
    pairs=defaultdict(dict)
    for g in B["groups"]:
        role=g["restart"]["role"]
        is_supported=int(g["state_cce_evidence"]["supported_prefixes"]==1)
        gate_by_role[role]["examined"]+=1
        gate_by_role[role]["supported"]+=is_supported
        rel=g["restart"]["scenario"]["related"]
        gate_by_history[str(rel)]["examined"]+=1
        gate_by_history[str(rel)]["supported"]+=is_supported
        pairs[g["restart"]["pair_id"]]["related" if rel else "unrelated"]=is_supported
    assert all(set(d)=={"related","unrelated"} for d in pairs.values())
    # Within-group realized terminal-return contrasts are descriptive; they are
    # not an independent counterfactual estimate of Qpi.
    action_alignment=[]
    for i in supported:
        eps=groups_B[i]
        keys=sorted({e["decisions"][0]["chosen_action"] for e in eps})
        realized={a:statistics.mean(e["reward"] for e in eps
                                   if e["decisions"][0]["chosen_action"]==a)
                  for a in keys}
        exemplar=credits[(i,next(e["sample_index"] for e in eps
                                 if (i,e["sample_index"]) in credits))]
        ev=exemplar["Qpi_evidence"]
        qpi={k:v["mean_policy_reward"] for k,v in ev["actions"].items()}
        gate=ev.get("signal_gate",{})
        action_alignment.append({
            "group":i,"role":B["groups"][i]["restart"]["role"],
            "pair_id":B["groups"][i]["restart"]["pair_id"],
            "related":B["groups"][i]["restart"]["scenario"]["related"],
            "sampled_first_action_counts":B["groups"][i]["first_action_counts"],
            "sampled_action_outcome_mean":realized,
            "frozen_policy_qpi_mean":qpi,
            "frozen_policy_gap":max(qpi.values())-min(qpi.values()),
            "frozen_policy_gate":gate,
            "chosen_actions_at_fork":keys,
        })
    dev=matched_compare(A["after"],B["after"])
    dev_from_base={
        "A_terminal":matched_compare(A["before"],A["after"]),
        "B_PCCE":matched_compare(B["before"],B["after"]),
    }
    train_distinct_first=sum(r["different_first_action"] for r in train_match.values())
    train_distinct_path=sum(r["different_full_native_action_path"] for r in train_match.values())
    group_metrics=[]
    for i in range(16):
        ag,bg=A["groups"][i],B["groups"][i]
        group_metrics.append({
            "group":i,"role":bg["restart"]["role"],
            "pair_id":bg["restart"]["pair_id"],
            "related":bg["restart"]["scenario"]["related"],
            "pCCE_supported":i in supported,
            "A_train_success":ag["safe_success"],
            "B_train_success":bg["safe_success"],
            "A_train_violations":ag["ci_violations"],
            "B_train_violations":bg["ci_violations"],
            "A_grad_norm":ag.get("pre_clip_gradient_norm"),
            "B_grad_norm":bg.get("pre_clip_gradient_norm"),
            "A_ref_KL":ag.get("mean_reference_kl"),
            "B_ref_KL":bg.get("mean_reference_kl"),
            "A_action_tokens":ag.get("trainable_action_tokens"),
            "B_action_tokens":bg.get("trainable_action_tokens"),
            "different_first_action":train_match[i]["different_first_action"],
            "different_path":train_match[i]["different_full_native_action_path"],
            "different_terminal_outcome":train_match[i]["different_terminal_outcome"],
        })
    stat={
        "study":"KCI matched16 CPU credit and action diagnosis",
        "inputs":{
            "backup":str(ARCHIVE),
            "SHA256":archive_hash,
            "archive_manifest_integrity":integrity,
            "frozen_inputs_state_pairs":8,
            "models_trained_on_cpu":0,
            "gpu_calls":0,
            "checkpoint_gradients_recomputed":False,
            "identical_initial_DEV_native_traces":True,
        },
        "head_to_head":{
            "A":{"safe_success":A["summary"]["dev_after"]["safe_success"],
                 "violations":A["summary"]["dev_after"]["violations"],
                 "safe_incomplete":A["summary"]["dev_after"]["safe_failure"],
                 "reward":A["summary"]["dev_after"]["mean_reward"]},
            "B":{"safe_success":B["summary"]["dev_after"]["safe_success"],
                 "violations":B["summary"]["dev_after"]["violations"],
                 "safe_incomplete":B["summary"]["dev_after"]["safe_failure"],
                 "reward":B["summary"]["dev_after"]["mean_reward"]},
            "preregistered_gate_passed":bool(matched["preregistered_exploratory_gate"]),
            "DEV_after_A_vs_B":dev,
            "DEV_change_from_identical_before":dev_from_base,
            "scenario_cluster_bootstrap_descriptive":bootstrap_scenario_clusters(A["after"],B["after"]),
            "train_same_seed_episode_different_first_action":train_distinct_first,
            "train_same_seed_episode_different_path":train_distinct_path,
        },
        "credit_mechanism":{
            "local_records":len(proofs),
            "supported_groups":len(supported),
            "role_counts":{r:dict(x) for r,x in roles.items()},
            "credit_sign_relation":dict(sign_counts),
            "mean_abs_terminal_adv_on_credited_first_decisions":
                statistics.mean(abs(r["terminal_standardized_advantage"]) for r in records),
            "mean_abs_local_addition_on_credited_first_decisions":
                statistics.mean(abs(r["weighted_local_addition"]) for r in records),
            "first_token_loss_coefficient_abs_ratio_on_credited":
                (sum(abs(r["weighted_local_addition"]) for r in records)/
                 sum(abs(r["terminal_standardized_advantage"]) for r in records)),
            "all_trainable_decision_tokens":all_branch_actions,
            "effective_loss_coefficient_abs_mass_terminal":proxy_terminal_abs,
            "effective_loss_coefficient_abs_mass_local":proxy_local_abs,
            "relative_extra_local_coefficient_abs_mass":proxy_local_abs/proxy_terminal_abs,
            "coefficient_mass_not_true_neural_gradient":True,
            "role_coverage":{k:dict(v) for k,v in gate_by_role.items()},
            "related_coverage":{k:dict(v) for k,v in gate_by_history.items()},
            "history_pair_supported_sides":{str(k):v for k,v in pairs.items()},
            "groups":group_metrics,
            "frozen_qpi_vs_realized_onpolicy_actions":action_alignment,
        },
        "limitations":[
            "Only one 16-step seed; 96 DEV trials from 12 source-adapted scenario conditions.",
            "The method adds extra frozen-policy rollouts; matching training groups is not equal inference cost.",
            "A single same-seed DEV run does not establish statistical superiority.",
            "CPU reconstructs signed scalar coefficients, not LoRA parameter gradients or action probabilities after updates.",
            "No independently new task family is evaluated; TRAIN states were selected from prior observed trajectories.",
            "Bootstrap over the 12 developed DEV scenarios is descriptive, not a generalization confidence interval."
        ],
    }
    OUT.mkdir(parents=True,exist_ok=True)
    results=OUT/"CPU_CREDIT_DIAGNOSTIC.json"
    if results.exists():
        raise FileExistsError(results)
    results.write_text(json.dumps(stat,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    detail=OUT/"CREDIT_DECISION_ROWS.jsonl"
    detail.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in records),
                      encoding="utf-8")
    h=stat["head_to_head"];c=stat["credit_mechanism"]
    lines=[
        "# KCI State-Directed PCCE vs GRPO: CPU-only diagnostic",
        "",
        "Input archive SHA256: "+archive_hash,
        "",
        "## Verified outcomes",
        "",
        f"- A terminal GRPO: success {h['A']['safe_success']}/96, violations {h['A']['violations']}/96, safe incomplete {h['A']['safe_incomplete']}/96, mean reward {h['A']['reward']:.6f}.",
        f"- B terminal+PCCE: success {h['B']['safe_success']}/96, violations {h['B']['violations']}/96, safe incomplete {h['B']['safe_incomplete']}/96, mean reward {h['B']['reward']:.6f}.",
        f"- Initial DEV full native action traces: identical; pilot superiority gate: {h['preregistered_gate_passed']}.",
        f"- After-training A/B DEV differs in {dev['different_full_native_action_path']}/96 action paths and {dev['different_terminal_outcome']}/96 terminal outcome labels.",
        f"- Paired A-only successes={dev['A_only_safe_success']}; B-only successes={dev['B_only_safe_success']}; A-only violations={dev['A_only_CI_violation']}; B-only violations={dev['B_only_CI_violation']}.",
        "",
        "## Reconstructed local policy-credit coefficients",
        "",
        f"- Accepted first-action credits: {c['local_records']}, across {c['supported_groups']}/16 TRAIN states.",
        f"- Same sign as terminal standardized advantage: {c['credit_sign_relation'].get('same_sign',0)}; opposite sign: {c['credit_sign_relation'].get('opposite_sign',0)}; near-zero terminal: {c['credit_sign_relation'].get('terminal_approximately_zero',0)}.",
        f"- Resulting first-action coefficient sign reversals: {c['credit_sign_relation'].get('net_advantage_sign_reversed',0)}.",
        f"- Mean |terminal standardized advantage| on accepted records: {c['mean_abs_terminal_adv_on_credited_first_decisions']:.6f}.",
        f"- Mean |0.5 x Qpi-centered local credit| on accepted records: {c['mean_abs_local_addition_on_credited_first_decisions']:.6f}.",
        f"- Extra local absolute coefficient mass / terminal absolute coefficient mass over all trainable branch tokens, using actual 1/(G x branches) objective weights: {c['relative_extra_local_coefficient_abs_mass']:.4%}.",
        "- This scalar mass is NOT a measured neural gradient ratio; true gradient alignment requires checkpoint/model computation.",
        f"- By role: {json.dumps(c['role_counts'],ensure_ascii=False)}.",
        f"- History-paired group support: {json.dumps(c['history_pair_supported_sides'],ensure_ascii=False)}.",
        "",
        "## Controlled A/B matched DEV action differences",
        "",
        f"- Identical seed & initial state per matched episode; changed first decisions: {dev['different_first_action']}/96.",
        f"- First role of diverging action: {json.dumps(dev['first_diverging_role'],ensure_ascii=False)}.",
        f"- B-A success rate cluster bootstrap (12 developed scenarios only): {h['scenario_cluster_bootstrap_descriptive']['B_minus_A']['success_delta']}.",
        f"- B-A violation rate cluster bootstrap (12 developed scenarios only): {h['scenario_cluster_bootstrap_descriptive']['B_minus_A']['violation_delta']}.",
        "",
        "## Interpretation limits and next diagnostic",
        "",
        "- At these fixed settings PCCE has no measured improvement over terminal-only GRPO.",
        "- We can identify whether additive local credits dilute, reinforce, or reverse normalized terminal credit; cannot infer they caused any particular episode change.",
        "- If most credits align but the effective objective mass is small, changing states/seed coverage and objective sensitivity may be more important than threshold adjustment.",
        "- If many credits conflict with terminal advantage, inspect whether this reflects valid delayed credit or teacher/rollout unreliability before retraining.",
        "- Next experimental claim requires predeclared independent task families and multiple seeds; do not equate TRAIN credit coverage with CI generalization.",
        "",
        "### Divergent DEV outcome examples",
        "",
    ]
    for d in dev["outcome_diff_examples"]:
        lines.append(f"- {d['scenario']} sample {d['sample_index']}: A={d['A']}, B={d['B']}")
    (OUT/"CPU_CREDIT_DIAGNOSTIC.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({
        "status":"CPU_DIAGNOSTIC_COMPLETE",
        "output":str(OUT),
        "manifest_integrity":integrity,
        "DEV_A_vs_B":dev,
        "credit_signs":dict(sign_counts),
        "role_credit_signs":{k:dict(v) for k,v in roles.items()},
        "effective_extra_coefficient_mass_fraction":c["relative_extra_local_coefficient_abs_mass"],
        "cluster_bootstrap":h["scenario_cluster_bootstrap_descriptive"],
        "gate_by_role":c["role_coverage"],
        "pair_support":c["history_pair_supported_sides"],
        "training_first_choice_disagreements":train_distinct_first
    },ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
