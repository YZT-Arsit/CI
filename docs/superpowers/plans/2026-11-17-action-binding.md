# Action Binding Implementation Plan

> For agentic workers: use superpowers:executing-plans inline. This is a staged research plan; later training choices depend on the recorded diagnostic and do not narrow the full objective.

**Goal:** Make relational information causally control the shared language policy's safe and useful actions, then verify genuine multi-step collaboration and source-grounded cross-task norms.

**Architecture:** Diagnose frozen policies before altering training. Separate relation computation, evidence-to-action readout, and environment execution. Use independently checked intervention pairs and immutable outputs.

**Tech Stack:** Existing Python / CPU PyTorch / pinned Transformers and PEFT in WSL; Windows UTF-8 orchestration; main branch.

**Spec:** docs/superpowers/specs/2026-11-17-action-binding-design.md

## Global Constraints

- Historical artifacts remain unchanged; no teacher evidence in final unassisted evaluation.
- CPU only; short forward/update batches, startup overhead separately recorded.
- All three user requirements remain open until directly verified.
- No retuning on old main tests, no successful-seed selection, no hidden invalid-action mask.

## Review Focus

- A route flip can be wrong on both sides; score both correctness and sensitivity.
- Forced answer prefixes and correct proof injection are diagnostic interventions, not policy success.
- Prompt/prefix token alignment, action order and action-token boundaries must be checked.
- A 32-edge graph does not establish 32 environment transitions or collaborative decisions.
- Benchmark-authored policies cannot be upgraded to externally validated business norms.

## Task 1: Frozen information-path diagnostic

Files: code/kci/action_binding_metrics.py; code/tests/test_action_binding_metrics.py;
artifacts/20261117_action_binding/{PLAN.json,build_diagnostic.py,probe_frozen.py,summarize_probe.py,LEDGER.md}.

- [x] Write tests for constant route, correct reversal, incorrect reversal, invalid output and misaligned pair IDs; observed seven assertion failures before implementing.
- [x] Implement pair metrics without access to model inputs or a runtime oracle; seven new tests pass, full current suite 187 passes.
- [x] Build the declared 24 grouped pairs across train/new DEV, preserving labels outside input and source hashes.
- [x] Freeze model/mode/data selection; run base/plain17/trace17/balanced17 under direct/gold/swapped evidence prefixes: 576 readouts sealed.
- [x] Check identity-input numerical control (zero difference), source/output hashes, full-vocabulary choices and teacher-forced bits. Input IDs have recorded hashes; full re-tokenization audit remains separate from the saved-record audit.
- [x] Summarize by model, split, depth and intervention; 192 additional same-DEV component interventions locate PATH dependence. Next experiment is independent candidate queries with/without paired margin, fixed in query_pilot/PLAN.json.

## Task 2: Repair the identified learning boundary

- [ ] Freeze a small, matched three-seed development comparison appropriate to Task 1.
- [ ] Test prediction-to-action coupling using correct, corrupted and self-predicted intermediates; if using internal interchange interventions, include identity and unrelated-variable controls.
- [ ] Confirm on new sealed groups, free generation and complete executions. Keep the direct supervised baseline, all failed seeds and resource budgets.
- [ ] Require input-level paired correctness plus causal use of learned relations before claiming P1 achieved; restore long-chain/norm/utility stress tests.

## Task 3: Genuine collaboration

- [ ] Select and pin an executable source environment; expose local observations and meaningful multi-actor state changes.
- [ ] Record separate relation-depth/decision/tool/message/transition counters; demonstrate 32 meaningful transitions without padding.
- [ ] Test missing, delayed and reordered messages and verify task/safety effects; evaluate the same learned policy with no forced successful actions.

## Task 4: Natural business policy scope and transfer

- [ ] Build a versioned clause-to-task source register, distinguishing real organization policy, benchmark policy and synthetic assumptions.
- [ ] Add independent label review and explicit UNKNOWN/exception/conflict cases.
- [ ] Test at least two task families with source-level holdout, applicable and inapplicable transfer; report policy validity separately from execution success.

## Task 5: Completion audit

- [ ] Review evidence for every spec requirement, retain unresolved items, run appropriate regressions and independent review before claiming the full goal complete.
- [ ] Publish reviewed progress on main with raw outputs, counts, provenance and limitations.
