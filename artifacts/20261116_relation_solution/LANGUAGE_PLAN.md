# Frozen language feasibility comparison

The 200-step joint pilot is preserved. Its headline 45/48 DEV Planner score is
NOT evidence of relational learning: many cases are ties and the output strongly
favours A. Audit found auxiliary yes/no labels shared A/B with action positions,
and safe questions were dominated by positive examples. Do not promote it.

Final comparison: SmolLM2-135M-Instruct pinned revision in source manifest; CPU
FP32 LoRA q_proj/v_proj r8 alpha16 dropout0, AdamW lr0.0008, weight_decay0,
gradient clip1, batch2. Seeds17/29/41. Exactly600 updates per arm, final checkpoint
only. `plain` predicts ANSWER only; `trace` predicts each candidate's immediate
safe flag, attainable safe-task flag, and a symbolic supporting path, then ANSWER.
Both use SAME paired state batches/renaming seeds and SFT objective; proof token
count differs and is reported. This is a learnability/repair diagnostic, NOT a
claim that SFT implements or outperforms the full proposed KCI DPO method.

Both members of a causal pair share the symbol vocabulary, option order and
prompt layout. Sample decision-state buckets by role/viability/unsafe alternatives.
All-optimal-only groups are excluded from policy training. For remaining ties,
the final choice loss sums probability over ALL optimal letters; it never forces
an arbitrary preferred action. All non-prompt target tokens, including EOS, are
trained. Gold paths/flags are TARGETS ONLY, never inference input.

The proof generator is an offline training teacher, using authorized relations
plus executed outcome labels. At inference the SAME pretrained LM generates its
own proof/answer freely. No exact closure, oracle action mask or fallback.
Invalid/missing ANSWER, generation timeout or capacity failure counts as failure.
Record generated proofs and independently audit their faithfulness; a correct
choice with a wrong proof does not establish faithful relational reasoning.

New main test: deterministic first16 base worlds of each frozen stratum, including
paired history swaps, unseen entity-word palette, depths4/6/10, noise, optional,
authorized, draft, both infeasible, broken/reverse/branch/cycle. Source holdout
10..17 is separate; record every incompatibility. Also compare seen-palette
predictions on the same base cases. No result-driven example selection.

The 128-word TRAIN and 128-word FRESH palettes are disjoint, deterministic from
the pinned tokenizer inventory; roles for each word are randomized per pair.
This is a compact structured-language finite-action task, not free-form agent
tool generation or independent validation of the added CI norms. All three
seeds/conditions and refusals must be reported. The >95% per-stratum gate remains.
