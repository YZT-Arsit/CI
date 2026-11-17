# Restore and audit the matched continuation

From repository root, restore prior dependencies as documented in `../20261117_frozen_crossover/REPRODUCE.md`. Restore this continuation's four archives with `python -B tools/restore_optimization_artifacts.py`.

`python -B artifacts/20261117_optimization_probe/audit.py` verifies17restored-step cases,68branches,51temporaryoptimizerupdates,no retained checkpoint,and170timedoperations. Base/tokenizer bytes and prior frozen adapters were verified by the original inference worker. This auditor makes no model calls.

`python -B artifacts/20261117_optimization_repair/audit.py` verifies the frozen worker/source/data/adapter hashes,all160trainingupdates,the exact original80batch schedule and teacher inputs,all448DEVepisodes,and both new models'320TRAINfitpresentations. It uses the previous native replay implementation under an explicitly redirected output directory. It reconstructs prompts,query/action parsing and environment transitions; no model calls are made. TRAIN oracle-action fit is reported separately from DEV self-generated actions.

Original commands,run sequentially in WSL,were:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_optimization_probe/run.py'
python -B artifacts/20261117_optimization_probe/audit.py
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_optimization_repair/run.py'
```

Both workers refuse an existing run plan. Fresh model replication requires a new declared directory and preservation of pinned initial weights,batches and source hashes. Do not remove original plans or overwrite recorded evidence. New adapters contain80additionalupdates; frozen reference adapters have zero added updates. The shared-model teacher-input loss is unchanged. The two continued models differ only in learning rate. CPUFP32,seed17,4threads;9.5shardguard per update;generation8ssoftlimit with>=9.5sinvalid.

The repair build preserves an initial version under prelaunch_revision. Before training, checks were added for TRAIN/DEV world-ID separation,history-pair strict reversal,and cross-view semantic identity. DEV uses new fixed seed350000000/350010000 but the same synthetic distributions and source task types; it is not independent-source validation. Local-step companions also remain TRAIN cases.

The separately qualified GitLab clause probe has its own `../20261117_grounded_norm_probe/REPRODUCE.md`. It uses a different frozen model and must not be pooled with these training results.

`audit_feasibility.py` is a label-only supplement added during inference:104of112presentations have a feasible compliant completion. `audit_residual.py` is a post-result inspection of the same8TRAINactionerrors;neither changes primary metrics. The separately registered `../20261117_option_fallback/audit.py` reconstructs216oracle-input option-order interventions on36TRAINpresentations and checks36identical-prompt controls. It is post-result diagnosis,not fresh validation or extra learned-policy evidence.
