# Restore and audit local supervision

Follow `../20261117_local_relation_step/REPRODUCE.md` to restore dependencies and this delivery's raw records. Then, from the repository root:

```powershell
python -B artifacts/20261117_step_learning/test_schedule.py
python -B artifacts/20261117_step_learning/audit.py
python -B artifacts/20261117_step_learning/training_check.py
python -B artifacts/20261117_step_learning/fit_probe/probe.py audit
python -B artifacts/20261117_step_learning/capacity_reference/reference.py audit
python -B artifacts/20261117_step_learning/capacity_format_repair/reference.py audit
```

These commands make no model calls. They rebuild all schedules and confirmation observations, verify sealed sources/data/checkpoints, check432 exact optimizer steps and2,992 new responses, and verify144 initial oracle responses reused byte-for-byte. The selected-TRAIN fit diagnosis adds288 new responses and zero updates. It uses fixed final checkpoints and cannot change the primary gates. The initial fit checkpoint sees all three selected panels; each trained checkpoint sees its own panel. Comparisons are within panel.

The Qwen reference adds24 CPU responses on metadata-selected identical user prompts, plus96 internal re-scorings of already-counted SmolLM outputs. It is a different pretrained/instruction-tuned model and numerical path, not a controlled model-size ablation. Its audit rebuilds the official chat rendering using the existing pinned tokenizer assets, verifies model identity, prompt token counts and resource release. It must not be presented as learned KCI policy behavior. Its native Windows launch was `python -B artifacts/20261117_step_learning/capacity_reference/reference.py run`, after both WSL workers finished.

All24 original Qwen responses were strictly invalid due to truncated explanatory output. `capacity_format_repair/PLAN.md` registers exactly one follow-up after observing this failure: same cases/decoding/scorer, append an explicit one-line/no-explanation instruction. Its24 additional responses are separate; no original scores or responses are replaced. The new prompts are not identical to the existing SmolLM prompts, so no extra SmolLM comparison is claimed. Its native launch was `python -B artifacts/20261117_step_learning/capacity_format_repair/reference.py run`. Both native audits run without model calls.

Original CPU worker commands, executed serially in the pinned WSL environment:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_step_learning/run.py'
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_step_learning/fit_probe/probe.py run'
```

Both refuse existing output directories. Do not delete evidence to rerun. A replication needs separately declared outputs. All three schedules share the same SmolLM2-135M initial LoRA checkpoint, CPUFP32/four threads, fresh AdamW lr0.0002 and144 updates each. The15second per-update watchdog is hard; greedy generation max_time8 and the9.5second scoring cutoff are not a hard process timeout.

`TOKEN_PREFLIGHT.json` was generated with the tokenizer only. It verifies bit/EOS target positions, paired token lengths, context bounds, and exact per-update token totals. The earlier audit-module import collision and its failed check are retained under `prelaunch_import_revision`; the final auditor uses an explicit file-path import. The source and plan were frozen again before training. No failed setup check is counted as a model experiment.

These are new-seed confirmations within developed structural families and vocabularies, not untouched task generalization. Controls repeat four structural templates; zero-noise groups repeat two graph templates. Oracle action retention uses correct external query inputs and is not autonomous policy performance. No model-generated multiround closure or32step Agent collaboration is established by these records.
