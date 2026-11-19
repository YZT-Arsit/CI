"""Summarize only independently audited completed CPU matched experiment.
Produces a negative-inclusive report; does not select checkpoints by DEV.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
S=HERE/"cpu_matched_01"
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def main():
    audit=read(S/"INDEPENDENT_AUDIT.json")
    assert audit["audit"]=="PASS" and audit["all_arms_same_start"]
    res=read(S/"RESULTS.json")["arms"]
    plan=read(S/"PLAN.json")
    pre=read(S/"TOKEN_PREFLIGHT.json")
    table=[]
    for arm in ("initial","paired","shuffled"):
        d=res[arm]["dev"]["summary"];n=res[arm]["native"]["summary"]
        table.append({"arm":arm,"individual_correct":d["individual_correct"],
            "flip_both_correct":d["flip_both_correct"],"hold_both_correct":d["hold_both_correct"],
            "native_query_correct":n["query_correct"],"native_action_optimal":n["action_optimal"],
            "native_immediate_unsafe":n["immediate_unsafe"],"native_invalid":n["invalid_actions"]})
    x={r["arm"]:r for r in table}
    # Strict diagnostic progress requires no regression of the original goals.
    gate={
      "paired_flip_strictly_better_than_both":
          x["paired"]["flip_both_correct"]>max(x["initial"]["flip_both_correct"],x["shuffled"]["flip_both_correct"]),
      "paired_hold_not_worse_than_both":
          x["paired"]["hold_both_correct"]>=max(x["initial"]["hold_both_correct"],x["shuffled"]["hold_both_correct"]),
      "paired_native_actions_not_worse_than_both":
          x["paired"]["native_action_optimal"]>=max(x["initial"]["native_action_optimal"],x["shuffled"]["native_action_optimal"]),
      "paired_native_unsafe_not_worse_than_both":
          x["paired"]["native_immediate_unsafe"]<=min(x["initial"]["native_immediate_unsafe"],x["shuffled"]["native_immediate_unsafe"]),
      "paired_native_queries_not_worse_than_both":
          x["paired"]["native_query_correct"]>=max(x["initial"]["native_query_correct"],x["shuffled"]["native_query_correct"])
    }
    success=all(gate.values())
    summary={"study":"32-step CPU LoRA batch grouping, seed 17","audit":"PASS",
             "model_start":plan["model_start_path"],
             "training_queries_identical_both_arms":True,
             "matched_actor_replay":True,
             "updates_per_treated_arm":plan["updates_per_arm"],
             "preflight_padded_query_tokens":{a:pre["training"][a]["total_padded_query_tokens"] for a in ("paired","shuffled")},
             "model_results":table,"diagnostic_gate":gate,"diagnostic_gate_passed":success,
             "gpu_training":"NO_GO","paper_level_method_claim":False,
             "limitations":["single training seed and developed synthetic two-task grammar",
                 "batch grouping changes padding, not strictly equal compute",
                 "binary per-candidate supervision, not new long-horizon preference optimization",
                 "fixed-state action evaluation, not autonomous full agent task",
                 "no external norm verification, not independent business transfer"]}
    (S/"DECISION.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# KCI-Align — bounded matched CPU LoRA study",
      "",
      "**Main research goal unchanged:** dynamic recipient-knowledge-conditioned long-horizon CI action policy learning.",
      "",
      "This is a small **batch-organization diagnosis**, not proof of a novel algorithm.",
      "One SmolLM2-135M base+LoRA start (archived truth17), 32 updates for each treated arm;",
      "the two arms see the SAME 128 native query targets and the SAME 32 archived actor replay batches.",
      "Only whether each flip/hold pair is colocated versus shuffled changes.",
      "",
      "| Arm | New DEV query /128 | Flip both /32 | Hold both /32 | Original query /192 | Original optimal action /80 | Unsafe /80 |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['arm']} | {r['individual_correct']} | {r['flip_both_correct']} | {r['hold_both_correct']} | {r['native_query_correct']} | {r['native_action_optimal']} | {r['native_immediate_unsafe']} |")
    lines+=["","## Predeclared diagnostic tradeoffs"]
    lines += [f"- {key}: {'PASS' if value else 'FAIL'}" for key,value in gate.items()]
    lines+=["",f"**Joint no-regression gate: {'PASS' if success else 'FAIL'}**.",
      "All published panels are **development diagnostics**; no independent held-out task family exists here.",
      "In particular, if flip performance improves while safety, hold, task utility or actions worsen, it is not a successful KCI repair.",
      "",
      "## Evidence and limitations",
      f"- The independent file/output/model auditor passed; raw outputs and adapter hashes are sealed in cpu_matched_01.",
      f"- Paired vs shuffled padded-query token counts: {summary['preflight_padded_query_tokens']}. Equal examples and steps; not equal FLOPs.",
      "- All gold labels come from the existing source-adapted deterministic environment, not a GPT self-judge.",
      "- The learned model has not completed the original 32-step multi-agent objective, and no external social norms were verified.",
      "- No GPU or paid API calls. Continue large-GPU NO-GO.",
      "",
      "## Reproduction",
      "Run Windows: py -3 artifacts\\20261119_composition_signals\\audit_cpu.py",
      "and: py -3 artifacts\\20261119_composition_signals\\analyze_cpu.py",
      "Do not rerun training over the same output folder. Preserve frozen checkpoints and all negative cells."
    ]
    (S/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"independent_audit":"PASS","joint_gate":success,"results":table,"gate":gate},ensure_ascii=False))
if __name__=="__main__":main()
