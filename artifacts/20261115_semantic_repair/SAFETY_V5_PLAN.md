# Restore immediate CI supervision without fabricating future preferences

Pre-result inspection found derive()/preference() suppresses all comparisons
whose winner cannot also succeed. That is appropriate for primary future
viability preferences, but discards immediate VALID > VIOLATION comparisons in
already-committed unsafe branches. Repair the training exporter only: keep
existing execution-derived pairs, and add immediate CI-dominance pairs with
their actual transition audits and exact values. No artificial preference among
equally safe failed outcomes. Existing environment/model code remains frozen.

Use the fixed V4 factorial worlds (same new-source holdout v3). Separate Worker
CI pairs from task-utility pairs in batching: Planner12, Source8, Worker CI4,
Worker utility8; non-Planner-only uses Source12, Worker CI8, Worker utility12.
Graph arms ci_only/combined/shuffled/no_relations: 1200 updates. Token arms
ci_only/combined: 6000 updates. Within each architecture all arms/seeds use the
same update budget, lr and architecture. Seeds17/29/41. Every chunk5 steps with
9.5s timeout; checkpoint every100 steps. All final checkpoints retained; no
test-based stopping, model choice or hyperparameter selection.

As with V4, the label ci_only means no Planner-future supervision, not a pure
safety-only objective. Token receives typed serialized triples, not free text.
No held-out model predictions have been inspected when this plan was written.
Do not release test results until V5 checkpoints are sealed too.
