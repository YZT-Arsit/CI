# Reproduction and evidence boundaries

Use the repository root and restore prior source/data archives first:

```sh
python -B tools/restore_semantic_artifacts.py
python -B tools/restore_relation_artifacts.py
python -B tools/restore_action_binding_artifacts.py
```

The last command restores169JSONL files using the published compression manifest. It verifies both compressed and original hashes and refuses to overwrite different evidence. A fresh isolated copy restored all169files, a second run restored0, and a deliberately different existing file was refused without overwriting it; see ARCHIVE_RESTORE_CHECK.json. Keep historic runs unchanged. Training/generation workers refuse an existing PLAN; replay in a new copy with explicit output paths, not by deleting plans in this archive.

The WSL CPU model environment and pinned SmolLM2-135M base are reused from `../20261116_relation_solution/lm_common.py`; follow that run's setup manifest. Base weights and installed dependencies are excluded from GitHub. This phase's saved adapters use the same 460,800 trainable LoRA parameters. All training in this phase is seed17 DEV work; multiple adapters do not mean multiple seeds.

Primary saved-output audits, with native Python from the root:

```sh
python artifacts/20261117_action_binding/audit_diagnostics.py
python artifacts/20261117_action_binding/audit_query_pilot.py
python artifacts/20261117_action_binding/audit_actor_transfer.py
python artifacts/20261117_action_binding/audit_secondary.py
python artifacts/20261117_action_binding/audit_learned_evidence.py
python artifacts/20261117_action_binding/audit_channel_and_capacity.py
python artifacts/20261117_action_binding/audit_learned_evidence.py coverage_repair
python artifacts/20261117_action_binding/audit_learned_evidence.py composition_repair_v2
python artifacts/20261117_action_binding/audit_learned_evidence.py risk_repair
python artifacts/20261117_action_binding/audit_alias_confirmation.py
python artifacts/20261117_action_binding/audit_risk_schedule.py
```

Run only completed stages; missing RESULTS means the stage is not complete. Audits reconstruct saved raw predictions and environment executions. Some write updated audit summaries or post-run integrity manifests; those are analysis products, not retroactive pre-run seals.

The alias-confirmation worker writes `RESPONSES_SEALED.json` before its separate scorer reads targets. It has16fresh world pairs and2dependent name/order renderings, not32independent tasks. `audit_alias_tokens.py` additionally uses the pinned tokenizer in WSL to check actual paired prompt lengths without loading a language model. Model-free loss checks are in `smoke_risk_loss.py`; they use the existing WSL torch environment and perform no optimizer updates.

Training dependencies: `query_pilot.py` supplies both the actor-transfer branch and the independent `learned_evidence.py` branch. The latter supplies `coverage_repair.py`; its coverage-only checkpoint supplies `composition_repair_v2.py`, while its class-balanced checkpoint supplies `risk_repair.py`. Each script's PLAN records actual data, initialization and schedule. The original `composition_repair.py` queue was cancelled before training because unequal group counts confounded depth and diversity; do not treat it as a model experiment. Source and dataset versions remain archived. `intervene_learned_evidence.py` is a same-DEV text-channel intervention, not an internal-activation or full-rollout experiment.

Updates have a 9.5-second watchdog; model load/setup is separate. Free generation has an 8-second soft limit and marks batches above 9.5 seconds invalid. Ollama capacity uses a separately declared 25-second request limit/60-second preload limit; actual timings and raw responses are retained, with `num_gpu=0`. Its 7B zero-shot results are not a controlled comparison of the trained135M method.

The tau2 subset is unmodified source from release `37199f36924c8896f5e048360691f8476cd89ba1`, with MIT notice and Git blob hashes. Native Windows execution requires `python -X utf8`; the first default-GBK failure is preserved. Install only the recorded small dependencies into an isolated target, then invoke `tau2_replay_utf8.py` in a fresh output copy. Its annotated-action oracle replay is not learned-agent performance or full policy compliance.

Business cases retain pinned GitLab source hashes, hidden-label review provenance and UNKNOWN cases. `policy_cases_v2/QUALIFICATION.json` scopes them to clause comprehension, not action-policy training or cross-organization applicability. Graph control cards record residual shortcuts and the required shuffled renderer. There is no large-GPU authorization gate pass in this run.
