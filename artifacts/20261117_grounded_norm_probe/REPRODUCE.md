# Restore and audit source-context clause predictions

Restore earlier source cases and the Qwen tokenizer assets through their existing helpers,then run `python -B tools/restore_optimization_artifacts.py` from repository root.

`python -B artifacts/20261117_grounded_norm_probe/audit.py` verifies the frozen source-case and label hashes,the shared context hash,exact84case/viewsequence,prompt reconstruction,official tokenizer count,raw response transport validity,strict/secondary parsers and model release. It reads labels only after response sealing. `python -B artifacts/20261117_grounded_norm_probe/check_parser.py` runs14format/negation/ambiguity controls without a model.

The original worker command is `python -B artifacts/20261117_grounded_norm_probe/run.py` on Windows. It uses the existing local Ollama service,requires no preloaded model,checks the exact Qwen digest,forces CPU inference and unloads its own model at the end. It refuses an existing run plan and does not retry failed requests. The root runner launches it only after the WSL optimization worker is terminal,avoiding concurrent heavy model jobs.

Contexts are authored paraphrases of the two pinned GitLab clauses,shared across all corresponding cases. Exact official source bytes were fetched from GitLab's API at the recorded revision and checked against the existing SOURCE_PIN hashes. Full policy text stays in `.cache/norm_probe_20261117`,not in this archive. The context review found and restored the conditional macOS deletion requirement before calls. No source labels were changed. IDs,condition groups and label/review metadata stay outside rendered prompts.

Primary responses must be exactly PASS,FAIL or UNKNOWN,apart from outer whitespace. The secondary parser additionally permits case normalization,an optional ANSWER: prefix and one final period/exclamation mark; it rejects explanations and negations. `prelaunch_revision` preserves earlier scorer/plan versions before this parser tightening and exact input-sequence/hash checks. All changes preceded model inference.

The original data card still applies:28authoredDEVcases,14conditiongroups,fourtaskwordings,twoorganizationclauses,model review only. Task paraphrases and answer-order rotations do not create independent policy sources. This is clause comprehension by frozenQwen,not transfer of the trained135Mmodel,not proof of actual business compliance,not human validation,and not execution of promised documentation/deletion duties. UNKNOWN remains unresolved.
