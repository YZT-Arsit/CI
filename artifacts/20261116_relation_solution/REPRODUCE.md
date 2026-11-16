# Reproduction and evidence boundaries

Restore JSONL first from the repository root:

```sh
python -B tools/restore_semantic_artifacts.py
python -B tools/restore_relation_artifacts.py
```

The new data generator is `code/experiments/relation_solution_data.py`. Training
worlds are the 288-cell factorial, two independent replicas, two matched history
variants per replica. Run the recorded `data_jobs.json` against a NEW output
directory; the scripts refuse to overwrite evidence. Raw source files retain
their original pinned URLs, SHA256 and original paths in `heldout_sources`.
Historical generator and failed-cache source versions are preserved under
`frozen_sources` with hashes matching the original job plans.

`DATA_AUDIT.json` records every dataset file hash and an independent directed-BFS
check of saved immediate-safety and safe-completion labels. `audit_data.py` can be
rerun without torch. The task laws are synthetic; PiSAs and MultiAgentPrivacy
provide computations, not independent validation of these added CI laws.

CPU environment: `environment.json`, `lm_install_report.json` and
`lm_source_manifest.json` record dependencies, native WSL paths and the exact
SmolLM2-135M-Instruct revision and file hashes. The 269 MB pretrained weight is
NOT included. `download_lm.py`, `setup_lm.py` and `profile_lm.py` document setup;
review machine paths before reuse. All model calls here are local FP32 CPU.
Ollama was available but is not the training backend for this run.

Frozen recipes: `PLAN.md`, `LANGUAGE_PLAN.md`, `structured_models/plan.json` and
`language_models/plan.json`. The structural control uses supervised primitive
gates/equality and an explicitly supplied max-min composition operator. It is
not free-text learning. The language comparison uses the same real pretrained
model, 460,800 trainable LoRA parameters, paired SFT, and either answer-only or
generated proof+answer targets. It does not establish full KCI-DPO efficacy.
`balanced_language_models` adds a DEV-motivated component-balanced SFT arm,
also three seeds and exactly 600 updates. `LANGUAGE_TOKEN_BUDGET.json` reports
target-token counts; equal update counts do not mean equal compute/token budget.

Train using `train_structured.py` or `train_language.py` in a NEW run copy after
updating its run paths. Files under the existing run are immutable evidence.
All updates/experimental chunks have a 9.5-second watchdog. Model download,
dependency installation and loading are separately measured setup operations.

Evaluation runners wait for checkpoint seals, verify hashes, and use no oracle
action masks or fallback. Primary language evaluation uses the first 16 worlds
per condition; structured evaluation uses all 64. The original external subset
contains only scenario 10 because the legacy adapter rejected seven sources.
`external_complete` preserves those failures and evaluates all eight sources
after adding the missing native constraint semantics, with unchanged models.
`evaluate_forced_states.py` separately intervenes on route choice to expose
dangerous Source/Worker states. These are diagnostic interventions, not naturally
occurring policy trajectories.
`representation_repair` keeps the local-rule weights fixed and changes the
random entity-code width/capacity. `repaired_runtime_evaluation` then tests all
semantic conditions on new seeds. The reusable `relation_runtime.choose_action`
uses this verified 64D/256-fact configuration. `auxiliary_ablation` and
`direct_pair_repair` are additional direct-learning diagnostics, not replacements
for the three-seed main result; both retain a collapsed seed.
The initial budget stops were superseded by the user's explicit time extension.
Their historical records remain under `deferred_before_user_extension`.
`extended_followups.py` resumes the language forced states; native transfer waits
for those to finish. `language_external_audits.py` and `balanced_audit_only.py`
complete the primary proof audits. No primary denominator or recipe changes.
`time_budget_probe.py` separately repeats the first two capped invalid batches
per proof model with a 12-second time limit and the same 256-token limit, prompts
and batch members. It never replaces the primary eight-second results.

`audit_executions.py` independently replays saved actions. It additionally checks
generated trace labels and every claimed path edge after the outputs have been
saved. `summarize.py` computes per-seed counts and paired two-sided success.
Invalid generation is a gate failure even if no private message was emitted.
Wilson intervals are descriptive; repeated seeds/variants do not create new
independent real tasks.
`audit_forced.py` independently checks every forced-state label and re-parses
its saved raw answer, then reports errors by decision role. `verify_final_run.py`
checks frozen checkpoint/data hashes and all sealed episode counters.
`audit_pair_decisions.py` separates first-route sensitivity to history swaps from
later Worker failures; it scores saved actions only and does not rerun models.
`PROMPT_PAIR_AUDIT.json` verifies equal input-token lengths, symbol vocabularies
and option order for 8,480 state/palette pairs. Reproduction of an audit should
write into a fresh run copy rather than overwrite its recorded timing evidence.

Regression modules run separately within 9.5 seconds using
`check_regressions.py`. One historical tokenizer module requires its original
Windows binary environment; its successful Windows receipt is retained beside
the failed WSL import. The new Source-encoding test runs with CPU torch in WSL.
The fresh Git archive verified 9,021 manifest files before the verification
receipts were added, restored both datasets, and passed the independent data
audit. Its first 9.5-second Windows-to-WSL test wrapper timed out. Resuming only
that step with a 30-second wrapper passed all seven new tests: 7.230 seconds
inside unittest, 21.007 seconds including startup/imports. This setup overhead
does not change training or primary inference budgets. Both the initial failure
and resumed receipt are preserved; `verify_publish.py` now allows this startup.

Preserved development failures: the early joint yes/no pilot's high DEV score
was dominated by ties; the first structured cache run used ragged arrays; the
first replay compared Python tuples to JSON lists. No failed output or model
prediction was rewritten. Later repairs have separate records.
