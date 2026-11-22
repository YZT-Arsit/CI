# KCI-Align: CI-first credit and natural-language observation, CPU follow-up 2026-11-22

## Research line remains Contextual Integrity

Dynamic recipient knowledge means a currently legal upstream disclosure can change future privacy consequences and safe task routes. Terminal outcome alone cannot distinguish a legitimate upstream action from a downstream agent's later violating disclosure. This led to two training mechanisms: CCE compares the continuation value of two currently legitimate actions at the SAME acting-agent state; DPR repairs the downstream agent's actual wrongful decision, without retroactively punishing the legal upstream disclosure. When exact ideal Qstar was found to disagree with the imperfect team Qpi, the implementation moved toward frozen-policy continuations and role-local parameter updates. Credit assignment is therefore a method for learning dynamic CI, NOT a generic new research direction.

## Structured truth versus model-visible text

The native environment, receipt log, privacy verifier, cost, task success and action IDs remain untouched. New tools/kci_llm_text_surface.py renders only the current actor-authorized observation as natural-language task/policy, actual delivered receipt history, publicly declared inference combinations, certified evidence bundles, resource terms and ordinary-language option descriptions. The policy returns ONE letter, mapped back to the unchanged executable native action. The renderer never supplies simulator hidden labels, Qstar or actual future outcomes. It is a faithful text-surface prototype rather than fully grounded enterprise prose. A GPT API may later vary wording but MUST NOT create task truth or CI labels. A and B preview: artifacts/20261122_kci_ci_first_sampled_gate_cpu_v1/ROLE_LANGUAGE_PREVIEW.md. All 28 existing worlds and all visited role decision stages passed an input/action-mapping smoke test.

## One targeted reward-to-credit repair

The previous native reward is successful return 1 - step_cost*steps, -1 on CI violation, 0 on safe incomplete. In the DG01 tradeoff task, a currently legal upstream package produced a slightly higher expected scalar value while the frozen downstream team's real CI violation probability was around 1.3%, vs near zero for the other package. At K=4, sampled continuations can miss this rare CI failure and label the higher-risk, lower-cost package as the preferred action.

New tools/kci_ci_first_preference_gate.py considers only publicly visible Horn inference combinations, actual R receipts, available package contents and required business checks. Future inferability is a caution signal, NOT proof that any current action is illegal. If K=4 continuation samples do not resolve a possible future CI risk that differs between two currently legal actions, an apparent cost-only advantage is no longer promoted to a DPO preference; the state is abstained. A witnessed native violation in the publicly hazardous branch can produce a safety-backed pair. No hidden native oracle Qstar/Qpi is inserted into prompts, the original terminal reward is unchanged.

The CPU ablation only changed the A-stage CCE preference gate; the R local repair and B CCE remain the same as in the existing 3-stage experimental script. Three seeds, equal K=4 per action, equal 80-stage updates, same 16 train /4 dev /8 developer-seen test.

| Metric (developer-seen exploratory worlds, 3-seed mean except counts) | Original scalar CCE | CI-aware A gate |
|---|---:|---:|
| Greedy safe task success | 8/8 in every seed | 8/8 in every seed |
| Ideal Qstar A strict-optimal actions | 6/6 every seed | 4/6 every seed |
| Ideal Qstar A regret per world | 0 | 0.007 |
| Exact native soft-policy CI violation probability | 0.005074 | 0.002249 |
| Exact native expected return | 0.636380 | 0.638670 |
| Accepted A preference pairs per seed | 11 | 10 / 9 / 9 |

Privacy risk fell by roughly 56% relative to the scalar preference version, and mean expected return rose slightly, with a clear tradeoff in ideal-Qstar optimal-action count. The removed DG01 tradeoff label originally favored a nominally lower-cost action despite uncertainty about downstream leakage. This is a nontrivial local reward-credit improvement; it is NOT independent LLM generalization evidence.

## Validation and files

Core source: tools/kci_llm_text_surface.py; tools/kci_ci_first_preference_gate.py; tools/run_kci_ci_first_native_gate_cpu.py; tools/audit_kci_ci_first_gate_cpu.py. Tests: code/tests/test_kci_llm_view_ci_gate.py (9 tests together with the two preceding relevant suites, all pass). Saved model audit reloaded 3 checkpoints and replayed 36 held-out developer-seen native episodes, 384 native action steps, 28 verified local downstream correction pairs; A training did not change B/R role logits. Source evidence: artifacts/20261122_kci_ci_first_sampled_gate_cpu_v1/RESULTS.json, INDEPENDENT_SAVED_MODEL_AUDIT.json and ROLE_LANGUAGE_PREVIEW.md.

The new language interface does not require API calls, any new synthetic data, a new normative truth system, or a rewrite of CCE/DPR. The next useful model-scale task is the first real pretrained-LLM role-text execution smoke and then small frozen-team counterfactual preference training, keeping the CI truth and reward evidence separate.
