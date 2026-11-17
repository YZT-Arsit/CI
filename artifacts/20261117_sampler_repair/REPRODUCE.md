# Sampler repair reproduction

Read `PLAN.md` and the frozen `PLAN.json` first. This is a2x2sampler/loss diagnosis, with two reused and two newly trained checkpoints, not four new training runs.

The data builder verifies8changed batch choices,72unchanged choices, identical view seeds, both task kinds, independent labels and history/view controls. The run additionally checks pure-data bucket reconstruction against the original training loader and hashes the original loss, initial adapters, data and baseline checkpoints.

Restore prior evidence and this continuation from the repository root:

```powershell
python -B tools/restore_relation_artifacts.py
python -B tools/restore_action_binding_artifacts.py
python -B tools/restore_binding_factors_artifacts.py
python -B tools/restore_fixed_state_artifacts.py
python -X utf8 -B artifacts/20261117_sampler_repair/audit.py
python -X utf8 -B artifacts/20261117_sampler_repair/summarize.py
```

`audit.py` reconstructs model-generated estimates, action prompts, parses, native states, outcomes and cross-view/history success for all four arms. Baseline copies retain their old checkpoint hashes and `ORIGIN.json` explicitly records zero new updates. The two repaired arms each have80training records and a sealed new adapter. No oracle estimate is provided to their evaluation.

For a new run, use an isolated full repository copy and new output directory, preserve the earlier archives and pinned CPU environment, and run `build.py` then `repair.py`. Never delete original PLAN files or overwrite evidence to rerun. `repair.py` is executed under the existing WSL Python environment, e.g. `wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_sampler_repair/repair.py'` in a fresh copy. Each update has a9.5second watchdog; original shorter inference bounds remain unchanged.

Changed batch composition affects query supervision, action preferences and consistency eligibility together. Do not interpret the result as isolating a single actor-gradient mechanism. Fresh random-world DEV is still generated from the same two task instances; it is not source-level holdout or a multi-seed confirmation.
