# Environment and data changes — 2026-11-12

Scope: bounded CPU prototype repair. The research objective remains recipient-knowledge-conditioned long-horizon CI policy learning. No LLM training, external API, or real-world privacy efficacy claim is introduced.

## Implemented fixes

- `known_to` now closes over recipient local/background facts and successful receipts the recipient is authorized to read. Undelivered records do not count. Reading a log does not grant access to the sender's source database or new transfer permissions.
- The common task prompt identifies the anonymous subject and the required private notification without naming the employee. Visible norms reflect the configured purpose and protected relations; action cost overrides are visible to the actor.
- Unknown recipients and unsupported routes return `INVALID_ACTION` before indexing state. Invalid/unknown transitions return independent state objects.
- Non-message actions reject nonempty semantic fields. `REQUEST_RULES`, `FINALIZE`, and `NOTIFY` have explicitly compiled message semantics. An absent body means the trusted structured-action renderer supplies the canonical message; a nonempty decision body must exactly match the structured answer or yields `UNKNOWN`.
- Requests, decisions, and notifications create actual delivered receipts. `FINALIZE` delivers the claimed result to A; `NOTIFY` delivers to the employee named in A's authorized identity fact. These are side effects of existing actions, so the default action costs remain 4 versus 7.
- Decision receipts have empty `fact_ids`: the actual declaration is retained in `body`, but hidden task correctness never changes visible fact metadata. This toy does not implement a general belief/claim logic for potentially incorrect decisions.
- The candidate enumerator covers bundled safe fact sends and initial A abstention, and uses the locally visible employee name. Stage restrictions reject redundant rule requests/computations and unsolicited rule sends. Exhaustiveness refers to this finite declared structured toy domain, never unrestricted natural language.
- Episode validation requires version/group fields, unique delivered event IDs, actual body, purpose, semantic verifier version, candidate binding, and bounded-search provenance. Statistical evidence statuses have an explicit schema. This is a structural gate; it is not a semantic, source authorization, or replay certificate.
- Preference exports include deterministic hashes of the prompt, chosen/rejected actions, and full record including evidence. Hashes bind bytes; they do not establish evidence truth.
- Grouped splitting supports explicit held-out rule families while keeping an entire base world together. Conflicting family assignments within one world are rejected.

## Parameter API for the bounded experiment

```python
env = ToyCIEnv(
    max_turns=9,
    facts=None,  # default seven facts; replace with an internally consistent mapping
    purpose='resource_planning',
    forbidden_inferences=None,  # default: R must not learn person_band
    task=None,                 # default anonymous task text
    action_costs={'SEND': 1.0, 'REQUEST_RULES': 1.0},
    world_id='base_world_001',
)
```

- `facts` retains IDs `identity`, `identity_other`, `band`, `rule`, `eligible`, `time`, and `person_band`. Each mapping key must equal the corresponding `Fact.id`. Facts are trusted structured inputs. Keep subject/value/text and the derived target mutually consistent.
- Entity and salary variants can use `dataclasses.replace`. The join matches identity.subject to band.subject and checks the person/band values against `person_band`; it does not hardcode Alice/B2. The task oracle remains the synthetic eligibility-yes task: use eligible B1/B2/B3 variants, not arbitrary new business rules or negative-eligibility claims.
- `forbidden_inferences=set()` is an explicitly authorized-combination positive control. Changing only the rule is observable in the normative context. It does not disable direct transfer permissions.
- Costs are finite nonnegative units by action kind; omitted kinds cost 1. Remaining action horizon is still measured in actions, independently of cost.
- For initial-background controls, construct `state = env.initial()` and set `state.local_fact_ids['R']` as required. For historical controls, append actual synthetic `Message` records with unique IDs, canonical body and explicit `delivered` status. Do not increment history by proposing candidates.
- `env.configuration()` returns the complete structured environment contract; `env.configuration_sha256` hashes it. `env.version` and initial norm/oracle versions bind to that configuration. Keep the configuration fixed during each rollout/comparison. Base-world identity is separately recorded as `world_id`/`scenario_group_id`.
- Shared receipt ACLs are explicit: R's inbox is readable by A/B/R; B's by B/R; A's by A. An inbox owner always knows its own delivered content.

Explicit family holdout:

```python
splits = deterministic_group_split(
    records, seed='2026',
    held_out_rule_families={'heldout_join_family': 'test'},
)
```

When a world has any held-out-family record, all its records follow that holdout. A single identity-band rule family does not establish rule-family generalization.

Statistical episode references use `status='estimated_preference'` (or `paired_statistical_estimate`) with `samples`, `delta`, distinct `sample_seeds`, `reference_policy_version`, `epistemic_status='statistical_estimate_NOT_optimality_proof'`, `comparison_metric`, and a two-number `confidence_interval`. Bounded references instead require `horizon`, `search_budget={'max_nodes': ..., 'max_actions': ...}`, and `generator_version`; successful references also require witness actions and cost.

## Changed files and verification

- `code/kci/toy_env.py`, `code/kci/candidates.py`, `code/kci/datasets.py` implement the above changes.
- `code/tests/test_env_data_regressions.py` adds 13 regression tests for the specific audited failures and new contracts.
- `code/tests/test_kci.py` updates the old expectation `known_to(R)=={'identity'}` to include R's initial rule, and enriches the schema fixture. Other legacy assertions remain unchanged by this work.
- `code/data/example_episode_contract.json` is still labeled `original_synthetic_fixture_not_from_pisas`; required provenance was added and its reference witness was recomputed using the current environment. The original is retained in the parent's baseline snapshot.
- The 13 targeted tests passed in 0.017 seconds in the first complete run. The default exact-cost regression still returns 4 and 7. Final targeted command receipts are in `env_validation.json`.

Remaining boundaries: exact canonical structured semantics, one deterministic inference family, three execution roles, synthetic fixed eligibility oracle, no natural-language verifier, no true shared-LLM learning, no external side effects. Configuration hashes, schema acceptance, or these CPU tests must not be promoted to empirical training or privacy results.
