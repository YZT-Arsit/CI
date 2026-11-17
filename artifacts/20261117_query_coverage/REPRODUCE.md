# Restore, audit, and reproduce

Restore prior dependencies through `../20261117_fallback_order/REPRODUCE.md`, then run `python -B tools/restore_query_coverage_artifacts.py`. The adapters use the already pinned CPU SmolLM2 source in `../20261116_relation_solution/lm_source_manifest.json`; model source weights are not duplicated here.

Original workers from repository root:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_query_coverage/probe.py'
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_query_coverage/run.py'
```

Workers refuse existing output. A fresh replication must use an explicitly declared separate output location and retain its own provenance. Do not remove existing evidence. The failed probe attempt is retained under `failed_probe_import`: generation of the first two-response batch completed, but the writer lacked the json import; no response text was retained. The fixed run changes only that import, and re-freezes the plan before any successful run outputs. These two unretained outputs are additional attempted inference, not valid scored evidence.

Audit existing records without model inference:

```powershell
python -B artifacts/20261117_query_coverage/audit_probe.py
python -B artifacts/20261117_query_coverage/audit.py
python -B artifacts/20261117_query_coverage/audit_pair_queries.py
python -B artifacts/20261117_query_coverage/audit_surface_controls.py
python -B artifacts/20261117_query_coverage/audit_lineage.py
```

The probe auditor reconstructs native reached states and original generation batch membership, then verifies that only completion-estimate bits change. The initial and corrected models reach different Worker states, so cross-model comparisons are not matched-state interventions. The paired within-model changes are the intervention evidence; no complete oracle rollout is run.

The training auditor verifies original TRAIN membership, non-depth factor alignment, side alignment by semantic labels, schedules, hashes, continuous optimizer steps, exact supplied teacher estimates, checkpoint identity, query prompts/labels, option permutations, and full autonomous replay. It reports fixed-panel query performance independently of on-policy queries because the latter can occur at different states after actions diverge. The fixed-panel contains every enumerated branch state for each DEV presentation, not just states reached by a particular policy. No panel label is used for training.

There are96jointupdates plus27fallbackupdates per trained arm. Same updates, roles, factor groups, labels and presentation seeds; differing depths imply different prompt lengths/token counts. Two frozen references prevent declaring a repair solely against the weakened corrected model. Preserve higher-depth, structural, oracle and complete-policy failure results separately. All32stepcollaboration/naturalnorm limitations remain.

`PAIR_QUERY_PLAN.json` was registered after main launch and before completed-arm metrics were inspected. It adds no model calls and does not change training or the primary gate. It scores all288native query pairs whose correct bits reverse, and144action-query groups across four presentations. These are correlated query groups, not144independentworlds. A pre-scoring review strengthened frozen source/data/dependency checks; the prior source/plan remain under `pair_query_pre_scoring_revision`.

Read SCOPE_CORRECTION.md before interpreting depth generalization. The initial trace ancestor already sampled depths1/2/3. This continuation restores their recent query/action coverage; it does not introduce depth3 for the first time in the whole training lineage. The original plan remains immutable and its broader wording is explicitly corrected.
