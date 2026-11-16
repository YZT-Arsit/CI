# Core evidence and comparison corrections

Date: 2026-11-12 (user workspace date). Scope: CPU-only prototype corrections; no model training, new loss, external API call, or research-goal change.

## Implementation changes

- `code/kci/contracts.py`: `State.signature()` now includes accumulated cost, final answer, and complete message event identity/delivery fields. This prevents a cached absolute trajectory cost from being reused for a state with different accumulated cost. `final_answer` is a trailing field for the environment's explicit decision-delivery contract. `CandidateGenerator` now declares its already-required `version` attribute.
- `RefResult` adds trailing `search_budget`; `CCEReport` adds trailing `reference_version`, `generator_version`, `search_budget`, and `horizon`, all with backward-compatible construction defaults. Missing provenance cannot certify a preference.
- `code/kci/continuation.py`: exact comparisons require the same state, actor, full authorized observation/Prompt, reference environment/generator versions, search budget, horizon, and recorded actual-policy snapshot. The embedded reference result must match the report envelope. CCE rejects a reference engine from a different environment version. Certified records carry reference versions, budget, horizon, and policy version.
- `code/kci/repair.py`: a legal, nonterminal prefix is not a failed trajectory. DPR now requires a confirmed violation or a verified terminal task failure. Unknown/invalid actions before the candidate failure keep the repair unresolved. Later observations do not undo an already confirmed violation.
- `code/kci/rewards.py`: `Outcome.safe_success` uses three-valued conjunction. A confirmed CI violation implies `safe_success=False` even when the stopped trajectory has `task_success=None`; the unknown task label is preserved. A later UNKNOWN does not erase an earlier confirmed violation. A nonterminal legal prefix does not establish full-trajectory compliance.
- `code/kci/interfaces.py`: the environment version contract explicitly covers fixed semantic configuration, including norms, observations, transitions, and costs.

## Conservative statistical contract

`PairedReferenceComparator` now preserves CI-first ordering instead of silently using safe-success probability as its substitute. It computes separate paired differences for full-trajectory CI compliance and safe success. For each complete dimension, the simultaneous two-interval Hoeffding radius remains `sqrt(2*log(4/delta)/n)`, conditional on independent seed draws and correct audits.

A statistical hard preference requires both:

1. A CI-compliance probability difference whose interval excludes zero.
2. At least one verified complete safe-success witness for the preferred branch, preventing unconditional refusal from receiving a certified positive preference.

Finite samples in which both branches always succeed do not prove equality of their population CI/task outcomes. Therefore safe-success differences and execution-cost differences remain diagnostics when CI ordering is unresolved. No self-reported Boolean equivalence flag or general proof system was added. Exact bounded reference search remains the implemented route to a certified cost preference.

`PairedSampleReport` keeps its original leading fields and adds separate CI diagnostics, known-pair counts, cost-pair count, and per-seed outcomes. Its `n` is the number of attempted pairs. An incomplete dimension has no unconditional mean/interval; known-only subsamples are not silently treated as unbiased samples. Conditional cost differences are now reported in actual environment cost units, with no asserted radius, because a step horizon is not a valid bound for arbitrary resource costs.

Certified statistical metadata includes policy/environment/norm/oracle versions, horizon, complete seed list and strides, per-seed CI/task/safe-success/cost outcomes, a replayable witness action sequence, the comparison scope, and statistical assumptions. The confidence statement is per comparison, not a global error guarantee over an adaptively selected training dataset. It remains an estimate for the fixed reference policy, not optimality evidence.

These corrections enforce the existing research priority `CI compliance -> task success -> cost`; they introduce neither a new optimization loss nor a new research objective. Proposal/SPEC source documents were not edited.

## Verification

New file: `code/tests/test_core_regressions.py` (16 meaningful regressions).

Coverage includes cache cost isolation; event/final-answer state identity; mismatched Prompt, reference version, generator version, budget, and horizon rejection; missing/stale provenance rejection; preserved exact 4/7 costs; rejection of partial/UNKNOWN DPR inputs; persistent known violations; known-zero safe success after leakage; no refusal preference; no sampled-all-success cost certificate; a synthetic case where higher CI compliance must override higher safe-success probability; and unresolved semantic outcomes.

The single old statistical-cost test in `tests/test_kci_extra.py` was updated, with explicit parent authorization, to expect `inconclusive` plus cost diagnostics. Its previous expectation depended on the invalid inference that sampled all-success outcomes establish upper-level equivalence.

All test processes were launched through `subprocess.run(..., timeout=10)` and used `python -B`.

- First integration attempt: 39 tests, one new test fixture failure. The environment correctly rejected an unsolicited R-to-B message before semantic checking; the fixture was corrected to use an ambiguous FINALIZE body. No production behavior was weakened to satisfy the test.
- Integration rerun before the final environment-version regression: 39/39 passed; unittest 0.331 seconds, enclosing subprocess 0.484 seconds.
- Final targeted core run: 16/16 passed; unittest 0.209 seconds, enclosing subprocess 0.361 seconds.

The parent run owns the final combined-suite verification after all concurrently edited modules settle. These are synthetic program/evidence-contract checks, not evidence of LLM learning improvement or real-world privacy safety.
