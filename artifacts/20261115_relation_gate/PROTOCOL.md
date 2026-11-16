# GPU prerequisite: relation and task-dependency falsification

Declared before running this suite, 2026-11-15 America/Los_Angeles.
The starting checkpoint is Git commit 507ddb7. Research objective is unchanged.
This is a falsification gate, not a promise of a positive result.

1. Freeze all 12 stage-3 finite-model checkpoints, feature extractor, and original
   task engine. No fitting, tuning, retries, or training on these evaluation cases.
2. Check actual dependency cuts: retain execution phase/budget, remove or change
   the prerequisite delivery immediately before its declared consumer. Report
   continued success, explicit rejection, and unhandled exceptions separately.
   Such cuts are fault injections, not claimed ordinarily reachable states.
   A task-irrelevant message must not be confused with a necessary input.
3. Replay both tasks under bijective entity/key renaming. Use identity, exact
   exchange of the previously correlated related/unrelated keys, and 32 fresh
   maps from seed 20261115. No role/action/field names change. Preserve character
   lengths, history counts, protocols, and equality/inequality relationships.
   Verify transformed full traces and protected relations, not just label copying.
4. Orthogonally rebind the next message's join key (two choices) and the actual
   history key (two choices), for each task and both budgets. Recompute labels
   by executing the finite engine. This 2x2 is a synthetic binding diagnostic,
   not four independent source tasks. A literal-history-only rule should fail
   while a relational diagnostic can distinguish matching from disjoint keys.
5. Exact optimal labels must be unchanged under global bijections. Missing any
   link, redirecting history away from R, or permitting the derived relation must
   remove the corresponding privacy conflict. Use a separately written join
   checker as an oracle diagnostic, never as a learned policy or training feature.
6. Enumerate safe successful paths; recompute costs under predeclared overhead
   weights 0, 1, 2, 3, 4 for private ACK/result relay and public sanitization steps.
   This is a sensitivity analysis, not a measured latency model. Fixed unit-step
   cost cannot establish natural task cost. Also use cost-free feasibility at a
   nonbinding horizon to distinguish privacy prohibition from schedule artifacts.
7. Learned-policy necessary gate: for EACH of the three history checkpoints,
   >=95% action accuracy and >=90% both-sides-correct rate on unseen renamings
   and orthogonal key binding; >=95% prediction agreement under valid renaming.
   These are finite-suite acceptance thresholds, not population confidence bounds.
   A failure is decisive for the current GPU readiness claim. Passing would still
   require independent source/structure holdouts and validation of synthetic norms.
8. Task dependency necessary gate: no required input can be deleted/redirected or
   replaced by an invalid delivered result while still reporting successful
   consumption. Rejection must be explicit; a KeyError is an engineering failure.

Record exact inputs, maps, actions, values, margins, paired outcomes and hashes.
Each experiment worker has a 9.5-second hard timeout. Missing/timeout results are
not a pass. No GPU or fresh LLM training. Do not rewrite historical results.
If a necessary gate fails, classify GPU training NO-GO and give the counterexample
and minimal repair requirements. Do not replace the learned model with a hand-coded
join algorithm and claim that relation learning was demonstrated.
