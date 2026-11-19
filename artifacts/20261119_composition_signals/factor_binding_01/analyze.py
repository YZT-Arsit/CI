"""Finalize evidence-based KCI factor-to-action decision only after audit passes.
No retry, no result-conditioned new experiments, no GPU or API.
"""
from __future__ import annotations
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def finish():
    audit=read(HERE/"AUDIT.json");plan=read(HERE/"PLAN.json")
    assert audit["audit"]=="PASS" and audit["same_initial_checkpoint"]
    stats=audit["results"];gate=audit["precommitted_gates"]
    arms=["initial","task_replay_control","relation_factor"]
    rows=[]
    for name in arms:
        x=stats[name]
        rows.append({
            "arm":name,
            "dev_task":x["dev"]["task"]["totals"]["correct"],
            "dev_relation":x["dev"]["relation"]["totals"]["correct"],
            "dev_task_flip":x["dev"]["task_pairs"]["flip_both_correct"],
            "dev_task_hold":x["dev"]["task_pairs"]["hold_both_correct"],
            "dev_relation_flip":x["dev"]["relation_pairs"]["flip_both_correct"],
            "confirm_task":x["confirm"]["task"]["totals"]["correct"],
            "confirm_relation":x["confirm"]["relation"]["totals"]["correct"],
            "confirm_task_flip":x["confirm"]["task_pairs"]["flip_both_correct"],
            "native_direct_correct":x["native_direct"]["query_correct"],
            "native_direct_optimal":x["native_direct"]["action_optimal"],
            "native_direct_unsafe":x["native_direct"]["immediate_unsafe"],
            "native_self_correct":x["native_self"]["query_correct"],
            "native_self_optimal":x["native_self"]["action_optimal"],
            "native_self_unsafe":x["native_self"]["immediate_unsafe"],
            "native_relation":x["native_relation"]["correct"]})
    outcome="PASS" if audit["all_gates_pass"] else "NO_GO"
    x={r["arm"]:r for r in rows}
    # This is a descriptive failure map, NOT an experimentally isolated mediation effect.
    if not gate["dev_relation_gain"]:
        root="Insufficient learned relation recognition on paired DEV under this training budget; no evidence for reliable autonomous knowledge update."
    elif not gate["dev_task_flip_margin"]:
        root="Relation may be improving, but reliable relation-to-task composition is not established by paired DEV judgments."
    elif not gate["original_native_action_retention"] or not gate["original_native_unsafe_retention"]:
        root="Local factor/task gains do not jointly preserve the native action safety-utility requirements."
    else:
        root="Insufficient joint evidence under the remaining precommitted retention gates."
    decision={
        "decision":outcome,
        "study_goal":plan["study"],
        "precommitted_gates":gate,
        "result_rows":rows,
        "failure_localization_hypothesis":None if outcome=="PASS" else root,
        "next_move":("Expand same hypothesis with seeds29/41, new business/task families, and real norms BEFORE large GPU" if outcome=="PASS"
                     else "STOP local loss/format/batching tweaks. Review data task realism and model capacity; do not scale factor-supervision without new independent evidence."),
        "confirmed_method_superiority":False,
        "confirmed_long_horizon_ci_policy_learning":False,
        "large_gpu":"NO_GO",
        "notes":["Synthetic root-disjoint DEV explored, not blind generalization.",
                 "Depth4 confirm shares source tasks and grammar; cannot claim new domain transfer.",
                 "One CPU LoRA start and 64 updates per trained arm.",
                 "Control extra objective repeats task answers; factor extra objective is relation truth. Padded tokens differ.",
                 "No 32-step effective autonomous agent collaboration or externally validated natural CI norms.",
                 "Self-generated factor route is observational prompt intervention, not causal proof of internal reasoning."]}
    (HERE/"DECISION.json").write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[
        "# KCI-Align Factor-to-Action Matched Study",
        "",
        "**Research direction unchanged:** dynamic recipient-knowledge-conditioned long-horizon CI strategy learning.",
        "",
        f"**Pre-registered joint gate: {outcome}.** Single seed, fixed model start, bounded synthetic source tasks.",
        "",
        "## Frozen paired comparison",
        "",
        "| Arm | DEV task/128 | DEV relation/128 | Task Flip/32 | Task Hold/32 | Relation Flip/48 | Depth4 task/192 | Depth4 rel/192 | Old native optimal/80 | Unsafe/80 | Self-factor optimal/80 | Self-factor unsafe/80 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['dev_task']} | {r['dev_relation']} | {r['dev_task_flip']} | {r['dev_task_hold']} | {r['dev_relation_flip']} | {r['confirm_task']} | {r['confirm_relation']} | {r['native_direct_optimal']} | {r['native_direct_unsafe']} | {r['native_self_optimal']} | {r['native_self_unsafe']} |")
    lines += ["","## Frozen gates (not changed after seeing model scores)"]
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name,value in gate.items()]
    lines += ["","## Mechanistic evidence boundary",
       root,"",
       "All relation/permission/required/utility labels were independently computed from the existing finite native task. Task-source documents and social CI norms are synthetic adaptations, not human validated. Relation evidence at inference comes from the model's generated text, not an injected oracle.",
       "",
       "This study does not show new long-horizon optimization or certified DPO performance. Pair supervision is a small component diagnostic, not yet an ACL method claim.",
       "",
       "## Independent checks and next decision",
       f"Audit checked raw outputs, adapter files and all three starts. {_bool(audit['same_initial_checkpoint'])}.",
       decision["next_move"],
       "",
       "All raw model responses and training traces are in this folder's study/ directory. Prior 20261117 archives are untouched. No paid API or large GPU used.",
       ""]
    (HERE/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"decision":outcome,"root_hypothesis":root,"gates":gate,
           "results":rows},ensure_ascii=False))
def _bool(x):return "Same model start verified" if x else "START MISMATCH"
if __name__=="__main__":finish()
