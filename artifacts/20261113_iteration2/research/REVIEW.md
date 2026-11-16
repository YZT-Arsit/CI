# Iteration 2 independent research and data review

The research objective remains unchanged: decisions must respect recipient knowledge accumulated through delivered history, while preserving a compliant path to the task result. This iteration supplies a bounded executable substrate and diagnostics. It does not establish a new learning method, an LLM improvement, natural-language CI judgment, or independent-task generalization.

## Findings that required correction

1. **Canonical source inconsistency (high severity, corrected in v0.2).** The v0.1 public constructor accepted a `Fact.value` change while preserving its displayed text. A payload could remain visibly 9 while the oracle used 1, reversing task preference at the same prompt. Factory-generated v0.1 cases did not contain this injection, but the public input contract was unreliable. `source_consistency_defect.md` preserves the exact counterexample; v0.2 source SHA256 is `9445b753f3cbf5d82db2871b5a830b54b7a84b8e2551b0484d8116b4306a997b`. The independent recheck rejected all 34 mutated source subject/text/value cases. This proves canonical typed consistency, not free-text semantic interpretation.

2. **Visible-prompt future-label aliasing (quarantined).** Re-enumeration of all 112 v0.2 files produced 3,552 safe nonterminal states and 3,152 exact prompt classes, of which 352 had multiple compatible states. Current-action CI labels and candidate menus were uniform. Safe future reachability and optimal cost disagreed for 48 prompt/action classes. Consequently, 96 strict preference signatures were not uniformly strict across compatible states and were quarantined. No actual reverse preference occurred in this subset: some states supplied a strict preference while another compatible state made both actions fail. These cases were ordinary final A notification states, not deliberately injected hidden-background examples. `observable_full_v2/` records stable signatures, raw IDs, and the disagreeing state evidence.

3. **History count shortcut (controlled by a separate matched release).** The first long-ledger fixture contrasts A SEND with A SKIP; with even heartbeat counts the receipt count distinguishes branches. It remains a valid environment/certificate invariance check, but cannot rule out a length shortcut. The separate `history_matched_v2` release uses A SEND on both sides and changes only a canonical join/scope key. The original release and its limitation remain evidence rather than being silently replaced.

## Current bounded checks

| Check | Independent evidence | Supported conclusion |
|---|---|---|
| Four declared inference structures | `independent_oracles_frozen_v2/`: 84 configurations and 432 primitive subset checks | Agreement with separately implemented finite closure conditions; reachable action grammar and exact costs agree in enumerated support |
| Rule boundary controls | `rule_counterfactual_audit.json`: 14 cases, including four constructor rejections | Disconnected joins, scope/unit changes, non-unique intersections and unsupported incomplete sources behave as declared |
| Source consistency | `source_consistency_recheck.json`: 34/34 rejected mutations | Typed source semantics cannot silently disagree with canonical displayed text through the audited constructor |
| Observable-label gate | `observable_full_v2/summary.json` | 96 unsupported strict preferences identified and retained in quarantine |
| Export and SFT controls | `dataset_policy_audit.json` | 4,860 raw preferences → 96 quarantined → 4,286 exact unique accepted; 1,982 SFT classes have a common minimum-safe-cost action across every compatible enumerated state; 1,170 classes excluded |
| Training isolation and optimization | `dataset_policy_audit.json`: all 40 jobs | No heldout family appears in training batches; no exact test-prompt overlap with train/SFT; same-fold/seed frozen references match elementwise; both U and E gradient checks pass, max absolute error 3.7180097889333297e-10 |
| Equal-length historical dependency | `matched_history_audit/`: all 80 groups, 160 records retokenized | Only one delivered receipt body changes; event count, character count, complete Qwen prompt token count match; independent declared closure flips protected conclusion true→false |
| Exploratory finite-policy history stress | `text_history_stress_audit.json`: 160 full-prompt features reconstructed and 40 frozen models recomputed | 800/1,600 side preferences correct, 0/800 joint flips, zero ties; max independently recomputed logit error 3.55e-15; N=0 also fails, so the result is not attributable specifically to long history |
| Exploratory lawful-prefix rollouts | `text_prefix_audit/summary.json`: 224 prefix states independently enumerated; 2,240 episodes / 6,720 policy actions replayed | Every forced A prefix is valid and ends at B, turn 1, cost 1; frozen full-menu argmax actions, transition verdicts, final costs and outcomes match saved records |

The reachable-state audits use an independently enumerated action grammar but execute the frozen production transition. They are not an independent implementation of every state transition. All source/family counts above describe this finite synthetic support.

### Lawful-prefix coverage result

Each arm includes 448 repeated-model episodes (112 existing source cases × two forced prefixes × two seeds), of which 392 have a safe successful continuation within the original horizon and 56 do not. The intervention is either actual A SKIP or actual A SEND of all context; the frozen policy then controls the complete action menu from B onward. Exact costs include the executed prefix, rather than resetting cost or horizon at B.

| Arm | Safe success | CI violation | Other task failure | CI violation among 392 feasible | Task failure among 392 feasible |
|---|---:|---:|---:|---:|---:|
| SFT-only | 120 | 112 | 216 | 56 | 216 |
| CI | 82 | 0 | 366 | 0 | 310 |
| Future | 72 | 112 | 264 | 56 | 264 |
| All | 60 | 112 | 276 | 56 | 276 |
| Balanced | 90 | 112 | 246 | 56 | 246 |

The 56 infeasible episodes per arm are not task-success opportunities; CI's total 366 task failures therefore includes 56 oracle-unreachable prefixes and 310 failures on feasible ones. Privacy violations remain violations even at an infeasible task prefix. Every observed violation in the other four arms occurs after forced context delivery, including 56 per arm for which a compliant continuation does exist.

The original 1,120 initial-state model rollouts all choose A SKIP first and show zero violations. This can be a valid safe initial-state strategy; the audit does not interpret safe avoidance as misconduct or evidence against anticipatory planning. It does show that initial-state rollout safety does not establish conditional B safety after a different lawful history. The forced-prefix results expose that narrower coverage gap. They are post-hoc diagnostics on the same cases and weights, not a new independent test distribution, a randomized estimate of learning benefit, or evidence about real LLMs. The arm counts should not be turned into a preferred-method conclusion.

## Boundaries required in reporting

- There is one shared workflow and one source/world construction. Four family-heldout folds test synthetic structural transfer within that workflow. Parameter values, names, horizons, authorization switches, history length and receipt placement are variants, not independent tasks. Alias chains are a multi-hop join family, not an independent application domain.
- The primary dataset assumes no hidden recipient task knowledge. Explicit hidden-background controls show that a same visible prompt may otherwise change CI validity. Such controls are excluded from hard-label training. The observable gate establishes consistency only over the declared enumerated compatible support; it cannot prove completeness over arbitrary external worlds.
- The 1,178 repair memberships contribute zero new unique pairs beyond CI/future enumeration: 1,132 are future+repair and 46 are CI+repair. The other accepted rows are 3,044 future-only and 64 CI-only. A balanced schedule changes weights; it does not isolate new repair information. Its 200 batches are 67 CI, 67 future and 66 repair, with overlapping membership.
- The finite policy consumes visible word/bigram hash features and a fixed 19-class canonical action menu. Its shared 24-unit tanh backbone, standard SFT and frozen-reference DPO are real small numerical optimizations, but they are not an autoregressive language model or LLM fine-tuning. Two seeds provide limited numerical sensitivity, not a population-level uncertainty estimate.
- Matched history controls are synthetic initial ledgers around one real A SEND action. The extra heartbeat receipts do not constitute executed agent turns, do not increase task-action cost, and do not show long-trajectory planning. Their equality excludes pure event/character/token-count explanations for the paired label flip. Fixed key values may still support lexical shortcuts. No model-use-of-history conclusion follows from the oracle controls alone.
- The additional frozen-model stress is an exploratory out-of-distribution test, not the originally predeclared main training result. It checks two candidate scores, not the full menu or an executed trajectory. All 40 models fail the joint historical flip even at the zero-heartbeat baseline. The 80 history groups are length/position renderings of eight family/answer cases, and scores across models are repeated observations rather than independent tasks. The largely unordered word/bigram representation does not test a real LLM attention or memory mechanism.
- All norms and source semantics are manually specified. `human_verified=false` and `natural_language_semantics_verified=false` remain required. Canonical rendering, exact inference, constructor rejection, and program checks do not substitute for independent human judgments or unrestricted natural-language understanding.
- The local batch-evidence wrapper spends predeclared pair/look budget before execution and preserves an interrupted reservation. The union-bound scope is one finite declared campaign, conditional on the underlying paired estimator assumptions. It is not anytime-valid or model-calibrated. Caller-provided policy/version identity must truthfully bind immutable code/weights; append-only API behavior is not tamper-proof storage. Hard-crash recovery was not simulated.

## Literature qualification

`source_review.json` records the local PDF hashes and the 11 selected pages reread from AgentCIScope, TG-CI, PrivacyFlow, ConCI, and the CI position paper. The useful reusable principles are source/task grouping, executable tool contracts, explicit visibility and delivered history, separated privacy/utility judgments, controlled norm/context changes, and bounded interpretation of policy compliance. These methodological precedents do not make the present hand-written workflow a reproduction of those task distributions or transfer their human validation to this dataset.

The earlier 29-paper extraction and selective method review remain in `artifacts/20261112_cpu_pilot/literature/` relative to the workspace root; this round did not reread all 29 papers. The independent audits above add no change to the research objective or core training loss.
