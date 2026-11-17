# Restore and audit

Restore prior dependencies using `../20261117_contrast_repair/REPRODUCE.md`, then from the repository root:

```powershell
python -B tools/restore_small_fit_artifacts.py
python -B artifacts/20261117_small_fit/audit.py
```

The native auditor requires no model inference. It checks frozen sources/data and parent seals, reconstructs the18selectedTRAINpairs and336prompts, verifies the exact144steps and loss choice per arm, compares initial parameter fingerprints and token exposures, rescoring every original/renamed query and oracle action. The initial oracle evidence must remain byte-identical to its source; it is excluded from new-call cost.

Original CPU worker:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_small_fit/run.py'
```

The worker refuses existing `fit` output. A new replication must declare a separate output location without removing this evidence. Model assets are the previously pinned SmolLM2 source in the relation-solution environment; archived adapters require that source. Training uses CPUFP32,4threads,144updates per arm,15second update watchdog. Generation uses max_time8 and a9.5second scoring cutoff, not a hard watchdog.

`prelaunch_review` retains the first frozen plan/auditor before two reporting clarifications. Only the audit counters and timing wording changed; data, schedules, checkpoint, training worker and dependencies were unchanged. Both revisions were reviewed before inference results were inspected. A shell quoting error in an import-only environment check occurred before Python could import the worker; the subsequent file-based WSL import passed with4threads. No model call was made by that environment check.

No full autonomous rollouts or independent-world generalization tests are performed here. TRAIN fit, same-world renaming and conditional oracle-action retention remain separate outcomes. Full-goal and GPU readiness cannot be inferred from a local pass.
