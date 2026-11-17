# Reproduce the frozen crossover diagnosis

Run from the repository root. Restore historical data/checkpoints with the existing helpers documented in `../20261117_teacher_binding/REPRODUCE.md`. Restore this run with `python -B tools/restore_frozen_crossover_artifacts.py`.

`python -X utf8 -B artifacts/20261117_frozen_crossover/audit.py` validates frozen inputs and checkpoints,regenerates native labels,reconstructs every query/action prompt,parses raw responses,replays all12state-policy combinations,and compares diagonal cells with previous online episodes. It additionally reconstructs the320TRAINpresentations and the13preflightresponses. It makes no model calls. Learned and oracle cells remain explicitly separate in RESULTS.

The original CPU inference commands were:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_frozen_crossover/run.py'
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_frozen_crossover/oracle.py'
```

The second worker requires the first worker's completed,verified learned-output seal. Both refuse existing run plans. Use a separately declared directory for fresh replication; do not delete plans or overwrite raw evidence. The worker checks base model/tokenizer bytes against the existing official source manifest,adapter seals,and imported helper hashes. Base weights and installed dependencies are not copied into the new archive.

`PLAN.json` and worker sources were frozen before main generation. `prelaunch_revision/` preserves an earlier plan and qualification/provenance fixes; the13-response preflight used identical generation logic before those fixes and is not independent evidence. The replay auditor was written/frozen during generation,before scoring,as declared in SCORING_PLAN. `scoring_revision/` preserves the earlier auditor before review added prior-file counts/hashes and the preflight hash. No model worker source was changed while running.

There are3querysources×560statepresentations=1680querymaps,containing3×1344candidatequeries=4032binaryresponses. There are9learnedactorcells×560=5040responses,3oracleDEVcells×560=1680responses,and2oracleTRAINcells×320=640responses. Reconstructed rollouts are112presentations×12cells=1344; only9cells use learned estimates,and only3diagonal cells use a single shared checkpoint. The320TRAINcases include repeated views and states. These are not new independent tasks.

Runtime dependency locations and package versions remain those in `../20261116_relation_solution/environment.json`. Batch4states,CPU4threads,FP32,greedy,max16newtokens,8ssoftgenerationlimit;elapsed9.5sorlonger marks outputs invalid. Data source and scope are recorded in PLAN.md and METHOD_SCOPE.md.

The TRAIN-query follow-up was registered before main output scoring in `TRAIN_QUERY_PLAN.json`. It freely generates on the same320originalTRAINquerypresentations for own_sft andteacher_sft:1600responses,160batches,no updates or new validation cases. Its original inference command was:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_frozen_crossover/train_queries.py'
python -X utf8 -B artifacts/20261117_frozen_crossover/audit_train_queries.py
```

The follow-up refuses an existing run plan. The second command only audits sealed responses and can be rerun after restoration. `COSTS.json` excludes this follow-up; its own generation counts and timing are in `TRAIN_QUERY_RESULTS.json`.

The first main audit stopped at native-label equality before reading model outputs: native candidate values were tuples, while JSON deserialization yields lists. `FIRST_AUDIT_FAILURE.json` records exact content equality after JSON roundtrip. `scoring_revision/audit_before_json_fix.py` and `SCORING_PLAN_before_json_fix.json` preserve that failed auditor. Only the equality comparison was normalized; the corrected scorer was re-frozen and then passed. No raw outputs, labels, or inference sources changed. `inspect_audit_failure.py` is a historical receipt generator, not a replay check: do not rerun it over the corrected auditor.

`audit_summary_support.py` checks finite label-summary compatibility without model outputs; `check_summary_support.py` tests its global-intersection logic against a three-way conflict and compatible controls. These diagnostics are not model learning or independent task validation.
