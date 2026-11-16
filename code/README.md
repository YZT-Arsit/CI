# KCI-Align v0.6 CPU audit — CCE & DPR algorithm-interface prototype

The 2026-11-12 CPU audit fixes knowledge/observation consistency, version-bound
preference certification, cache isolation, terminal traces, and failure-only DPR.
See [the implementation amendment](../docs/KCI_Align_CPU_Implementation_Amendment.md)
and [the pilot report](../artifacts/20261112_cpu_pilot/REPORT.md).
The original v0.5 code and documents are preserved under the pilot's `baseline/`.
The research objective is unchanged. Real LLM training and independent-world
evaluation are **not ready**; renamed toy copies are not independent data.

**Purpose.** Make the candidate action generator, continuation search, verified preference exporter, and downstream policy repair inspectable and runnable. This repository is a **research prototype**, not a reproduced paper result or a trained LLM. The included world is small, synthetic, deterministic, and uses exact structured facts; no third-party data is copied.

## Run without external dependencies

Python >= 3.10. Run from package root:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python examples/run_demo.py
```

The demo writes `data/verified_demo_preferences.jsonl` using exactly the prompt/chosen/rejected/action-metadata interface needed by a DPO pipeline. The reference model (`pi_C`) used for search is **not** the frozen DPO reference (`pi_0`). Do not confuse them.

## Important results from the toy environment (NOT empirical claims)

The task is to determine the eligibility of an anonymous employee and privately notify the employee. A owns the real identity, B owns the exact salary band, R owns qualification criteria.

- If A does **not** give R the identity mapping at the beginning, R may receive the anonymous band, complete the decision, and have A privately notify Alice. Finite exact reference cost: **4 actions**.
- If A first gives R the identity mapping, sending the anonymous band afterward would reveal a forbidden person-to-band relation. B can request the criteria from R, compute the eligibility locally, share the authorized result, and let A notify. Finite exact reference cost: **7 actions**.
- A deliberately broken B policy ignores recipient history and sends the band anyway. The CCE reports `new-unauthorized-inference:R:person_band`. DPR constructs a *same-state* repair: B should request criteria instead.

These counts are fixed by this hand-written toy task. The demonstration shows that the **interfaces and evidence gates function correctly**. It does **not** demonstrate learning improvement, generalization, method novelty, or real-world privacy safety.

## Files / interfaces

| Module | Public interfaces | Purpose |
|---|---|---|
| `kci/contracts.py` | `State`, `Observation`, `Action`, `StepAudit`, `RefResult`, `Preference` | Serializable semantics / evidence status |
| `kci/interfaces.py` | `CIEnvironment`, `SemanticVerifier`, `TaskOutcomeVerifier`, `ExternalPolicy` protocols | Replace toy world with real agent task environments |
| `kci/candidates.py` | `CandidateGenerator`, `ToyCandidateGenerator`, `JSONCandidateGenerator` | Generate candidate actions using **only** authorized observation |
| `kci/toy_env.py` | `ToyCIEnv` | Deterministic multi-agent delivery, knowledge closure, rules, task evaluation |
| `kci/continuation.py` | `ReferenceSearch.search`, `ContinuationEvaluator.evaluate`, `certify_preference` | CCE and certified action comparisons |
| `kci/repair.py` | `DownstreamPolicyRepair.repair` | DPR, verified same-state corrections |
| `kci/rewards.py` | `Outcome`, `summarize_rollouts` | Separate CI status, task success, and cost |
| `kci/datasets.py` | `audit_pisas_files`, `validate_enriched_episode`, `export_verified_preferences` | Prevent treating static annotations as dynamic RL rewards |
| `kci/paired.py` | `PairedReferenceComparator.compare` | Empirical paired-seed comparison for frozen non-exhaustive reference policies; not optimality proof |
| `data/example_episode_contract.json` | Sample enriched dynamic episode | Illustrative schema and validator input |

### The five core Python calls

```python
from kci import ToyCIEnv, ToyCandidateGenerator, ReferenceSearch
from kci import ContinuationEvaluator, DownstreamPolicyRepair, certify_preference

env = ToyCIEnv()
gen = ToyCandidateGenerator()
reference = ReferenceSearch(env, gen)
cce = ContinuationEvaluator(env, reference)
s = env.initial()

# A normal action and another legal option from the same pre-action state:
r1 = cce.evaluate(s, env.act('A', 'SKIP'))
r2 = cce.evaluate(s, env.act('A', 'SEND', 'R', ('identity',)))
pair = certify_preference(r1, r2)  # returns None if evidence is inadequate

# Downstream repair requires an actually observed failed Trace:
repair_engine = DownstreamPolicyRepair(cce, gen)
# repaired = repair_engine.repair(failed_trace)
```

## Rules that guard reward quality

1. **Do not give the generator oracle data.** Its interface accepts an `Observation`, and the observation must be authorization-filtered. In production, explicitly supplied CI norms should be visible as task context; hidden ground-truth labels should not.
2. **Proposed data do not become knowledge.** Only successful delivery events update the recipient's ledger. External state cannot be mutated by evaluating uncommitted candidates; CCE uses cloned states.
3. **Verify the actual language text.** The toy only accepts exact canonical structured output. A production integration needs a model-independent semantic extractor/entailment verifier with `UNKNOWN` status for ambiguous or unsupported claims. Never trust model-reported `fact_ids` without checking its message body.
4. **Do not award false feasibility.** Finding a valid success continuation proves *a path exists*. It does not establish that the found path is best. An LLM candidate generator always reports `exhaustive=False`.
5. **Cost-based future preferences require strong proof.** The included code produces an exact cost preference only when the finite declared action space and transition model have been fully explored within the horizon; otherwise it returns `None`, unless there is a verified feasible path versus certified bounded-horizon infeasibility.
6. **Do not reward unconditional refusal.** To prefer a safe action over a violating action, the safe action must have a verified continuation that actually finishes the task. Legitimately impossible tasks require a separate verified-decline contract; the prototype excludes these pairs rather than making up rewards.
7. **Do not blame the upstream actor for the downstream semantic violation.** DPR starts at the actual failed step and tests same-observation alternatives. Earlier legal actions can receive independent future-path preferences if the reference comparison is reliable.
8. **CI policy remains contextual.** Sender, recipient, purpose, fact content, and inferred protected relations must be checked. A repeated direct forbidden disclosure is still a violation even when the information was already known.
9. **Keep the training objective lexicographic in the core specification:** CI compliance → verified task success → execution cost among valid successful trajectories. Scalar penalties are postponed until the learning algorithm needs them.
10. **Group data splits by underlying world + rule family.** Changes of names, ID values, message order, and paraphrases from the same case must not leak between train and test.

## Evidence statuses

- `exact_within_horizon`: only for the complete, deterministic finite declared action/transition domain. Not global optimality.
- `proven_unreachable_within_horizon`: only if **all declared** actions were exhaustively examined with the bounded step horizon. Not a proof about the unlimited real world.
- `verified_witness_only`: one safe successful continuation found, but the candidate set, solver budget, or semantics is incomplete. Supports **existence**, not a cost ordering against another incomplete candidate.
- `unknown`: no supported conclusion. Do not generate training negatives from this status.

The `evidence_status` is carried into the training metadata. In a realistic LLM task, expect many continuation outputs to be estimated or `unknown`, not exact.

## Connection to the user's CI paper corpus and publicly released data

| Data source | Fields/concepts verified in actual public files | Safe reuse in KCI | What must be newly annotated |
|---|---|---|---|
| **PiSAs** | `scenario.json` task/cast/org/timeline; `utility.json` task oracle and decision rule; `appropriateness.json` attribute-level labels; `visibility.json` recipient visibility | Business context, roles, static norm evidence, task correctness; prompt/role scaffolding | Delivery ledger, derived-fact CI norm, counterfactual branch state, action viability, reference continuation |
| **MultiAgentPrivacy** | Distributed per-agent tables, compositional inference steps, benign and sensitive goals | Relation/composition templates; distributed fact structure | Full CI authorization/context, legal alternative actions, task reward and history-conditioned supervision |
| **CIDER** | Individual contextual disclosure granularity and user decisions | Optional disclosure granularity and human-preference context | No native multi-agent delivery ledger or dynamic verified continuation |

**Important:** A PiSAs `timeline` entry is a pre-existing artifact, **not an agent-to-agent delivered event**. `audit_pisas_files` returns `train_ready=False` and describes new fields required. `validate_enriched_episode` rejects a fake `delivered=True` field without the rest of a proper event record, but complete semantic checks still require replaying the event and norm inside an environment adapter.

Sample public PiSAs files used for schema inspection:

- https://huggingface.co/datasets/ServiceNow/PiSAs/blob/main/meeting_allocation/scenario_01/scenario.json
- https://huggingface.co/datasets/ServiceNow/PiSAs/blob/main/meeting_allocation/scenario_01/utility.json
- https://huggingface.co/datasets/ServiceNow/PiSAs/blob/main/JIRA_allocation/scenario_01/scenario.json
- https://github.com/ServiceNow/PiSAs/tree/main/scenario_generation/notebooks
- https://github.com/Vaidehi99/MultiAgentPrivacy/tree/main/data

To audit a local PiSAs case: `PYTHONPATH=. python examples/audit_pisas.py /path/to/scenario_01`.

## Remaining integrations (explicitly unfinished)

- Load a real LLM inference backend (e.g. Qwen) into `JSONCandidateGenerator`; verify free-form `body` text independently before CI scoring.
- Implement an environment adapter for the chosen natural-language multi-agent workflow: tool ACLs, actual message delivery, task state transitions, deadline and success oracle.
- Implement an inference checker for natural-language entity linking/composition with provenance; use human adjudication for ambiguous normative cases.
- For non-exhaustive reference rollouts, `PairedReferenceComparator` requires a confidence-supported **CI-compliance probability advantage** plus a verified safe-success witness for the preferred branch. Safe-success and cost differences remain diagnostics: finite samples do not prove equal higher-priority risks. Exact bounded search remains the route to certified cost preferences. Per-comparison error bounds do not automatically control a large batch's overall error.
- Connect the exported `prompt`/`chosen`/`rejected` action JSON to a TRL DPO training loop with a **frozen separate** `pi_0`. No GPU training is run here.

See [SPEC_v0.5.md](SPEC_v0.5.md) for exact operator and data matching contracts.
