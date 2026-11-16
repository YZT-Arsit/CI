# Active Stage 3 handoff — NOT final report

User authorized one hour starting 2026-11-13 08:39 UTC, deadline ~09:39 UTC.
Full active goal is available from get_goal. All five requirements remain in
scope; do not mark complete based on this partial checkpoint. Current time
~08:49 UTC, approximately 50 minutes remain. No live experiment processes.

## Authoritative completed work

- Read Proposal v0.6 and existing paired CCE implementation. Old comparator only
  certifies CI-rate advantage; success/cost are diagnostics without upper equality.
- Added `code/kci/stratified.py`: frozen finite paired-outcome population,
  predeclared n/delta/cost bound, simultaneous three Hoeffding intervals, strict
  priority. Exact finite-support upper equivalence certificate enables sampled
  success or cost differences. Requires all supported rollouts successful for
  conditional-cost tier. Unresolved outcomes remain UNKNOWN. This is only a
  calibration bridge, NOT scalable arbitrary-LLM non-exhaustive CCE.
- Six new `code/tests/test_stratified.py` tests passed, 0.152s. Dataclass field
  annotations added after first test/calibration; rerun final checks after new code.
- `code/experiments/stratified_calibration.py`: 7 controls x certificate on/off x
  8 seeds =112 comparisons, n1024, delta .01 EACH. No family-wide guarantee.
  8/8 success and cost preferences with certificate; without certificate UNKNOWN.
  Tiny adverse risk blocks cost; null, unknown, mixed-success cost unresolved.
  Raw outputs/receipts and `cce_calibration_summary.json` exist. These are outcome
  table tests, not real task transition cross-calibration yet.
- `code/experiments/future_signal_audit.py`: 112 workers re-evaluated legacy
  prefixes/actions with protected-derived-fact permission enabled, keeping task
  and candidate actions unchanged. Max .310s, all complete.
  Joined all retained contributing raw IDs to 4176 released future pairs:
  ordinary task reachability3926, ordinary compliant cost226,
  CI-mediated task reachability12, CI-mediated cost12. No mixed contributors.
  **This permission intervention proves no history-conditioned flip.**
  `legacy_signal_categories.jsonl`, `legacy_signal_summary.json`, raw worker files.
- New files only; prior artifacts untouched. No relevant memory hits. ARS
  experiment workflow read and announced. Latest developer disallows proactive
  delegation except explicit user/skill request; no subagents invoked in this stage.

## Required remaining work

1. At least two distinct executable dynamic workflows (source-inspired, clearly
   adapted synthetic norms), genuine history-caused future value/action flips.
2. Integrate CCE with actual task rollout populations, frozen reference and exact
   continuation cross-calibration. Do not pretend finite table certificate solves
   general non-exhaustive CCE; state boundaries and unknowns.
3. Full visible-observation equivalence audit; isolate conflicts; authorized result
   receipts may expose actual task completion, no evaluator-truth leakage.
4. New finite policy and frozen local Qwen targeted short diagnostics, DPR unique
   state/pair counts and overlaps. Real LLM parameter training deferred.
5. Versioned data/contracts/license distinctions, tests, replay, factor coverage,
   negative results, final requirement-by-requirement completion audit/report.

## Source/task design options considered, NOT implemented

Prior sources at iteration2/source_anchors. Meeting fixed original task solver
is in iteration2/source_task_probe/task_probe.py (`meeting_problem/check`),
3375 schedules ->6 without typed private constraints ->1 with all constraints.
Jira625->2->1; MultiAgentPrivacy source has unspecified income/debt thresholds,
so any precise replacement thresholds must be explicit adaptation, never original.
PiSAs default no-memory preload exposes entire artifacts; attribute visibility
is judge ground truth. Source manifests/cards/ACL_CLARIFICATION are unchanged.

Potential distinct dynamic workflows:
- Meeting negotiation and reservation: current public alias roster vs private
  booking request both legal; later HR alias-case disclosure combines with
  recipient's earlier case-cause history and roster into forbidden named medical
  fact. Related vs unrelated legal history changes cheap raw continuation vs
  multi-step sanitized continuation; private booking has intermediate service ACK.
  Unit-cost executed steps can yield fast3/private4/sanitized5, producing strict
  cost reversal and, at horizon4, reachability difference. Booking must use actual
  original task constraints and a real terminal reservation/result receipt.
- Relational query/aggregate workflow inspired by MultiAgentPrivacy, distinct
  data-shard/reduce state topology and exact declared thresholds. Current action
  opens raw join vs trusted aggregate path; existing recipient mapping or
  complementary totals alters future derived CI. Need genuine task computation,
  messages, and result verification, not identical old workflow under new names.
These are options, not commitments or evidence. Keep contracts minimal and make
all synthetic additions/permissions/source differences explicit.

Existing legacy classes (State/Action/Observation/ReferenceSearch) are usable;
inspect authoritative code before choosing integration. Full old99 tests should
remain unchanged. Use existing `experiments.run_short_jobs` with unique manifest
names and <=9.5s workers; it records logs, source hashes and failures.
