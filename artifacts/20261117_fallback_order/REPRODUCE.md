# Reproduce and audit the matched-order pilot

Restore previous dependencies through `../20261117_fallback_weight/REPRODUCE.md`, then run `python -B tools/restore_fallback_order_artifacts.py`. The initial model is the same sealed `optimization_repair/repair/continue_lower_lr` used in the prior weight experiment, not any newly selected winner.

Original worker from repository root:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_fallback_order/run.py'
```

The worker refuses an existing `repair` directory. Fresh replication requires a separate declared output directory; never remove or overwrite this evidence. Each learned arm gets27fixed correction batches and80originaljointbatches, exactly once each. Only the merge order differs. All format targets are fixed independently of global step: correction targets are selected by immutable caseID; replay loss receives its original batch index. CPUFP32, seed17,4threads, lr0.0002, freshAdamW per arm, no internal reset, clip1,12second update guard. Each generation retains the earlier9.5second validity rule.

`correct_then_replay/midpoint` has27correctionupdates; `replay_then_correct/midpoint` has80jointupdates. They are not matched-compute checkpoint comparisons. Midpoints are diagnostic only and were never eligible for best-checkpoint selection. Trainable parameter/Adam-state digests before and after each diagnostic must match. Final checkpoints each have107updates. The frozen initial reference has no new updates.

```powershell
python -B artifacts/20261117_fallback_order/test_schedule.py
python -B artifacts/20261117_fallback_order/audit.py
python -B artifacts/20261117_fallback_order/verify_replay_control.py
python -B artifacts/20261117_fallback_order/audit_planner.py
python -B artifacts/20261117_fallback_order/audit_failure_locations.py
```

The native auditor checks frozen sources/data/assets, exact schedules, optimizer step continuity, teacher inputs, correction target ties, equal per-batch token counts across arms, intermediate/final weights, and evaluation identity receipts. It replays every final free-policy episode and reconstructs all old/fresh oracle permutation prompts. It reports paired presentation-level and all-six-case transitions, not just aggregate midpoint differences. Native tuple values are JSON-normalized before comparing saved labels.

Core suite: from `code`, `python -B -m unittest discover -s tests -v`; retained output is `CORE_TESTS.log` (187tests,1skip,exit0). The schedule test first failed because the interleaved candidate concatenated all replay batches; after repair it verified matched multisets, stream ends, and block boundaries.

This experiment keeps the original shared-model goal. Oracle-action diagnostics are not autonomous-policy results; source-adapted synthetic workflows are not independent business-norm validation; three policy decisions are not32stepAgentcollaboration. The original108correctionprompts are repeated renderings of7distinct zero-completion observations from5basegroups. Report every failed family and gate.

`planner_probe.py` is a separately frozen, post-launch diagnostic registered after the first blocked arm showed successful oracle Worker binding but weak autonomous outcomes. It runs only after the main seal exists, on allfourfinalcheckpoints and all112initialPlannerstates per checkpoint. Only the estimate values change; observation, names, options and model remain fixed. Original command: `wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_fallback_order/planner_probe.py'`. This adds448oracle-input action responses and no updates, not448episodes or complete oracle rollouts. `PLANNER_PROBE_PLAN.json` freezes sources and interpretation gates; previous integrity-check revisions are retained. The native auditor separates80unique-optimum states from32tied states and reports identical-prompt controls. This diagnostic does not change the original interleaved repair gate.

The post-launch repeated-prefix control compares the reverse arm's first80jointupdates with the earlier `fallback_weight/uniform` checkpoint and its216TRAINresponses. It verifies frozen prior evidence before comparing weights and outputs. This is a reproducibility control, not independent generalization.
