# KCI-Align — Native Composition Signal Audit (2026-11-19)

**Research objective unchanged:** learning long-horizon CI action policies conditioned on evolving recipient knowledge in multi-agent systems. This is a bounded **data and failure-diagnosis experiment**, not a new method or GPU-ready dataset.

## Sources and limits
- Reuses native receipt_dataset.make_world, relation_solution_data.targets, ReceiptTask, and relation_language.prompt.
- All labels derive from deterministic native transitions and exact finite continuation. Zero GPT API calls.
- Source task families remain Meeting and Query, from pinned source adaptations; CI rules are synthetic, not externally validated.
- One paired group contains two history counterparts of the same task, authorization, required-output and delivery conditions.
- Existing frozen experiments, training data, weights and scoring rules are untouched.

## Generated data
Files are under pilot_01:
- worlds.jsonl: 128 synthetic worlds from 64 paired roots.
- queries.jsonl: 1536 native per-candidate task-success questions.
- pairs.jsonl: 768 paired history comparisons; 64 flip and 704 hold.
- quality.json and manifest.json: quality counts, factor coverage and hashes.
- shortcut_report.json: deliberately history-blind baseline.
- balanced_train_candidate.jsonl: 128 TRAIN-only examples (32 flip pairs and 32 hold pairs; 64 positive, 64 negative). NOT training approved.
- curriculum_report.json: sample coverage and provenance.

TRAIN and DEV each have 32 roots, 768 questions and 32 flip pairs. Each split: Planner 8 flip, Source 8, Worker 16. This is root-disjoint but not independent task/grammar generalization.

## Crucial shortcut finding
A supervised frequency lookup using role, action class, task kind, graph depth, permission, Full-required, and delivered flag, but **never historical edges**, scores:
- 736/768 (95.83%) individual DEV correctness
- 0/32 DEV flip pairs both sides correct
- 352/352 DEV hold pairs both sides correct

Therefore overall CI/utility accuracy is severely confounded by easy invariant cases. Evaluate both sides of controlled flip/hold pairs, separated by role and safety/utility condition. Do not interpret high average accuracy as knowledge-state reasoning.

## Validity and research gate
- Native replay and integrity auditing pass.
- Seven regression tests cover replay, file tampering, split integrity, flip-per-role coverage, prompt/label isolation, shortcut baseline and curriculum balance.
- New questions are BINARY TASK VERDICTS, not certified same-state DPO chosen/rejected action preferences. training_ready=false is intentional.
- Do not compare these new labels with archived v1 DPO data as if same protocol or supervision.
- No model improvement, autonomous 32-step task, external norm validity or independent business transfer is established.
- For the next minimal model comparison, hold model starts, tokenizer, total updates, original-query replay, action replay, and retention controls constant. Compare the same verified target labels under ordinary versus paired composition sampling and preserve all failures.
- DEV is explored. It is not a fresh blind TEST.
- Keep large GPU NO-GO until model, action and external-generalization gates are met.

## Reproduce on Windows (no API, no GPU)
From D:\research\longhistory:

py -3 artifacts\20261119_composition_signals\test_signals.py -v

py -3 artifacts\20261119_composition_signals\build_signals.py audit --output artifacts\20261119_composition_signals\pilot_01

Rebuild only into a fresh directory; no overwrite:

py -3 artifacts\20261119_composition_signals\build_signals.py build --output artifacts\20261119_composition_signals\another_run --repeats 2

Then run shortcut_audit.py and make_curriculum.py on the new output.

**Decision:** continue bounded model diagnostics, not broad synthetic-data growth. No paid API or GPU in this stage.
