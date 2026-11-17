# Reproduce and audit the fallback weight pilot

First restore the earlier archives as described in `../20261117_optimization_repair/REPRODUCE.md`. Then run `python -B tools/restore_fallback_weight_artifacts.py` for this run's raw JSONL records. The base model/tokenizer revision is inherited from `../20261116_relation_solution/lm_source_manifest.json`; the initial checkpoint is the sealed `optimization_repair/repair/continue_lower_lr`, not a freshly initialized model.

Original worker command, from repository root:

```powershell
wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_fallback_weight/run.py'
```

It refuses an existing `repair` directory. Fresh replication must use a separate declared output directory, preserving the evidence here. CPU FP32, seed17, four threads, fixed160totalupdates, same80batches per newarm, lr0.0002, freshAdamW, clip1. Uniform multiplier1 and weighted multiplier4 differ only in the semantic actor loss. The denominator is a fixed full-schedule mean, not per-batch normalization. The frozen initial arm receives no updates. No early stopping, best-checkpoint selection or output-dependent reruns.

Audit without model calls:

```powershell
python -B artifacts/20261117_fallback_weight/test_weights.py
python -B artifacts/20261117_fallback_weight/audit_native.py
python -B artifacts/20261117_fallback_weight/audit_breakdown.py
python -B artifacts/20261117_fallback_weight/audit_loss.py
```

The original `audit.py` remains frozen. A reviewer identified that native candidate values/relations are tuples while stored JSON values are lists. `audit_native.py` round-trips only reconstructed native targets through JSON, then runs the unchanged auditor. `SCORING_CORRECTION.json` freezes this correction during uniform training, before DEV scoring. It does not change labels, metrics, model calls or losses.

The auditor verifies input/source/checkpoint hashes, replays all free-policy episodes, reconstructs all query/action prompts, checks all160trainingrecords, verifies the global weight mass, and reconstructs all oracle diagnostic prompts and tied optimal-action sets. Fresh states are selected without model outputs from the same synthetic DEV worlds; six permutations per state are dependent. The previous216TRAINprompts are regression evidence only. No oracle estimates enter the free-policy evaluation.

The unit test first failed the expected4:1weight ratio under uniform weights, then passed after the weighting change. `IDENTITY_CHECK.json` additionally verifies the real mixed-Worker loss is unchanged at multiplier1. Core suite: from `code`, `python -B -m unittest discover -s tests -v`; raw output is retained in `CORE_TESTS.log`.

The protocol does not test real32stepAgentcollaboration or naturalbusinessnormtransfer. Passing its local gate cannot authorize largeGPUtraining or establish the full research goal.

Post-result probability diagnosis: `loss_diagnostic.py` was frozen separately in `LOSS_DIAGNOSTIC_PLAN.json` before it ran, after main inference finished. Original command was `wsl.exe -e bash -lc 'cd /mnt/d/research/longhistory && python3 -B artifacts/20261117_fallback_weight/loss_diagnostic.py'`. It uses all108TRAINzero-completion oracle-order prompts and the three frozen checkpoints, records full-vocabulary probabilities at the action position, and checks agreement with already sealed free generation. It refuses existing probability records. There are81forwardbatches,324conditionalrecords and zero updates; these are not additional free responses. `audit_loss.py` reconstructs the probability-gap counts and verifies their hashes. `audit_breakdown.py` separately reports actual violations and invalid episodes without changing the frozen gate.
