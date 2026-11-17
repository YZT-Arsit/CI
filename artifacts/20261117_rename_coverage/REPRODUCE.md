# Restore and audit

Restore dependencies through `../20261117_contrast_repair/REPRODUCE.md`, then run from the repository root:

```powershell
python -B tools/restore_small_fit_artifacts.py
python -B tools/restore_rename_coverage_artifacts.py
python -B artifacts/20261117_rename_coverage/test_views.py
python -B artifacts/20261117_rename_coverage/audit.py
python -B artifacts/20261117_rename_coverage/localize.py
python -B artifacts/20261117_rename_coverage/width_control.py audit
```

These commands make no model calls. The main auditor reconstructs all data and labels, verifies frozen sources, dependencies and checkpoints, checks exact update schedule/teacher inputs/token totals, rescores every query/action and independently replays all autonomous transitions. It verifies480 reused control responses byte-for-byte and excludes them from new generation costs. `localize.py` adds descriptive failure and task-solvability counts; it does not change the main gate or establish causation.

The tokenizer-only preflights were run in the pinned WSL environment, with no model loaded. `TOKEN_PREFLIGHT.json` checks every update's aggregate token count; `PROMPT_PREFLIGHT.json` checks1920 individual training prompt/answer comparisons and84 six-prompt evaluation groups. Both include script and plan hashes. The first token-preflight attempt incorrectly called the environment dictionary and failed before tokenizer loading; the failed script/log are retained under `pre_repair`. They are not experimental outputs.

Original CPU worker:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_rename_coverage/run.py'
```

The worker refuses an existing `study` output directory. Do not remove preserved evidence to rerun it. A new replication needs a separately declared output location. It uses the previously pinned SmolLM2-135M model and LoRA assets, CPUFP32/four threads,144 updates,15second update watchdog, generation max_time8 and9.5second validity cutoff. The latter is not a hard watchdog. Fixed control inference is rerun only for the new panels/worlds; its earlier480 outputs and adapter are reused with receipts.

`copy_sources.py` adapts the prior world builder only by changing seed ranges, and native replay only by changing the entity palette. `rename_data.verify()` verifies those exact transformations. The third palette is new relative to the two registered experimental palettes, not claimed unseen in base-model pretraining. Fresh generated worlds use previously developed task families. No external task generalization, natural norm validation or32step collaboration claim follows.

The separately registered batch-layout calibration follows `WIDTH_CONTROL_PLAN.md`. Its original launch was `python -B artifacts/20261117_rename_coverage/width_control.py build` after main sealing, followed by `width_control.py run` in the same WSL environment. Both commands refuse existing outputs. It generates672 additional responses, records both final checkpoint hashes and unchanged decoding settings, and keeps all primary results/gates intact. The native `audit` command above compares bytewise and parsed outputs, timing/invalid counts and initial padded widths without inference.
