# KCI executable synthetic data and frozen policy evaluation (development release)

Date: 2026-11-19. This is a synthetic multi-agent Contextual Integrity research sandbox. People, sensitive relations, company/customer records, and privacy norms are fictional. Business workflow patterns are inspired by AgentDojo Workspace and tau2 Retail; these are NOT official upstream benchmark tasks or scores.

## Source and reproducibility

- code/kci/knowledge_world_seed_v2.py, public_workflow_ci.py, public_workflow_ci_grounded.py and public_workflow_ci_remaining_grounded.py: evidence-grounded synthetic tasks covering W01-W05 and R01-R05.
- code/kci/diverse_ci_branch_env.py: seven pairwise non-isomorphic logical knowledge hypergraphs, three different business flow families and separate recipient R/intermediary M ledgers.
- code/kci/diverse_ci_long_horizon_env.py: irreversible privacy-preflight planning action, deadline cost and real downstream business branch.
- tools/build_diverse_ci_long_horizon_dataset.py and audit_diverse_ci_long_horizon_dataset.py: reproduce and independently replay 21 structural task roots, 189 worlds, 539 observations, 105 contrasts and 168 trajectories.
- tools/evaluate_frozen_kci_ollama.py, audit_frozen_kci_ollama_results.py and check_frozen_kci_action_order_bias.py: actual local frozen LLM actions with native outcome replay.
- For original detailed experiments, see the README inside the following three synthetic-data archives.

Selected archived research assets under artifacts/20261119_composition_signals (explicitly committed despite other artifacts being ignored):

- KCI_ALL10_GROUNDED_WORKFLOWS_V1.zip
- KCI_DIVERSE_KNOWLEDGE_BRANCHES_LONG_HORIZON_V2.zip
- KCI_FROZEN_QWEN25_7B_DIVERSE_EVAL.zip

### CPU test commands

    py -3 -m unittest discover -s code/tests -p "test_diverse_ci*py" -q
    py -3 -m unittest discover -s code/tests -p "test_public_workflow_ci*py" -q
    py -3 -m unittest discover -s code/tests -p "test_knowledge_world_seed*py" -q
    py -3 -m unittest discover -s code/tests -p "test_dynamic_tasks.py" -q

71 such tests passed on the authorized Windows machine. Native audits replay 189 worlds, 539 model observations, 105 paired contrasts and 168 trajectories.

## Local CPU-controlled policy study

For 63 synthetic business conditions:
- No early privacy preflight, local greedy reaction: safe success 28/63
- Always open private channel: safe success 56/63
- Business/authorization-aware rule controller: safe success 63/63
- Bounded exact simulator search: safe success 63/63

These are deterministic simulator-policy reference results, not trained model scores. The strong hand-written controller solves the whole current synthetic set.

## Frozen local LLM study (NOT Qwen3-4B)

A local Ollama qwen2.5:7b-instruct-q4_K_M 4-bit model, digest 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e, was run without training, updates or paid API calls:
- Seven knowledge rule families with five historical/authorization variants: 35 source-disclosure probes. All 35 indexed-action choices selected SEND_RAW_R, 7 incurred immediate CI violations, correct Flip response 0/7.
- Position bias check: 4/4 action-list swaps changed semantic choice while the model kept outputting action_index=0. Hence indexed source probes are confounded by answer placement; they are not conclusive pure CI reasoning measurements.
- Four additional direct-named-action checks still selected raw disclosure. Full episodes use named actions.
- Eight full business trajectories with 62 actual model actions: 0 safe successes, 3 CI violations, 5 safe incomplete. All were independently replayed against the native state and business environment. Same 8 conditions: rule controller and bounded oracle each succeeded 8/8.
- The former remote 5090 GPU/Qwen3-4B checkpoint was unavailable. Local Qwen2.5-7B q4 results cannot be represented as the original Qwen3-4B or as model-size scaling evidence.

## Required research improvements after development snapshot

P0: Establish a real long-horizon action-value preference Flip **due solely to previous delivered receiver knowledge**, with business state, permissions, budget and tools fixed. An immediate CI violation flip is insufficient.

P0: For the same legitimate visible state, branch over two candidate actions, run real downstream continuations and save true terminal privacy, utility, cost and sampling uncertainty/provenance. Successful vs failed independent sampled trajectories do not automatically identify Qpi preference.

P0: Keep strong baselines: always open, never open, business-rule-only, immediate CI greedy, explicit planning, and fair same-observation source access. Preserve the 63/63 symbolic strong baseline.

P1: Test relevant/unrelated/authorization histories, vary true tool preconditions, recipients and legitimate alternative downstream actions, and preserve genuine held-out independent task mechanisms.

P1: Separate training-valid preference labels, complete RL trajectories, single-step labels, and eval-only counterfactuals. These test cases are developer-inspected, not blind benchmarks.

Status: DEVELOPMENT SYNTHETIC DATA + ACTUAL FROZEN BASELINE FAILURE. NOT A PAPER-GRADE BENCHMARK, NOT A NEW LEARNED RL MODEL AND NOT A CERTIFIED HUMAN SOCIAL NORM DATASET.
