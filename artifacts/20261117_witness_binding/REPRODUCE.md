# Restore and audit copied-witness supervision

Restore dependencies through `../20261117_step_replay/REPRODUCE.md`, then use the delivery restore script and native checks from repository root:

```powershell
python -B tools/restore_witness_binding_artifacts.py
python -B artifacts/20261117_witness_binding/test_graphs.py
python -B artifacts/20261117_witness_binding/test_scores.py
python -B artifacts/20261117_witness_binding/data_audit.py
python -B artifacts/20261117_witness_binding/audit.py
python -B artifacts/20261117_witness_binding/diagnostics.py
python -B artifacts/20261117_witness_binding/training_fit/audit_fit.py
```

These checks make no model calls. The main auditor reconstructs the480-query confirmation and every192-step schedule, verifies paired initialization, exact targets/token totals/loss decomposition, six final adapters, all outputs and432byte-identical reused Worker responses. It separates binary behavior, valid generated witnesses and conditional action retention. No generated witness is substituted for truth or supplied to another model call during evaluation.

Original worker, in the pinned CPUFP32/four-thread WSL environment:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_witness_binding/run.py'
```

It refuses an existing `study` directory. Preserve evidence; a replication requires separately declared output paths. Each paired bit/proof continuation begins at the same corresponding prior replay adapter, with fresh AdamW lr0.0002 and192updates. The proof arm adds auxiliary examples and compute; it is not token/compute matched. Update watchdog15seconds is hard; generation max_time8 and9.5second score cutoff are not hard process timeouts. All planned conditions remain, with no score-based stopping or retries.

`DATA_AUDIT.json` distinguishes the8/12/16node generation domain from actually printed nodes: unused isolated nodes are not serialized. World IDs and new seeds do not establish independent business tasks or untouched structural generalization. The random graph panel is label-balanced rejection sampling inside explicit size/density/frontier cells; the oracle is used to construct/evaluate data, not as runtime input.

Worker retention is repeatedly used development evidence with correct external estimates; Planner/Source/global-query retention and autonomous task success remain untested here. A valid proof string is not established as the causal basis of the binary answer. Natural norms, UNKNOWN repair and32step Agent collaboration are separate unresolved goals.

The separately declared `training_fit` diagnostic evaluates the first12original batches perseed after main checkpoint sealing:144selected TRAIN presentations,9models×48queries×2modes=864newresponses. It changes no training, gates or checkpoint selection. Run `training_fit/run_fit.py` only after a completed main seal and only in a fresh declared output location; it refuses existing outputs. Its auditor checks original prompt/name/order identity and frozen model parameters. These scores must remain separate from confirmation.
