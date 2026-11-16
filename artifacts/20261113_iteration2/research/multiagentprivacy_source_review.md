# External source anchor: MultiAgentPrivacy

Read-only review of the official `Vaidehi99/MultiAgentPrivacy` repository at commit `6e7af5d8a8b8a803d8d51cefc01831d2e0724b41`. Coverage: repository root listing and README; complete `data/experiment_10_def_adv_both.json`; complete `scripts/eval_leakage_exec.py`; selected sections of `scripts/qwen_gemini_map_cot_sensitive_set_hist.py` (281–368, 388–448, 640–707, 952–1006, plus function/history-location screening). No execution, model/API inference, dataset-wide review, or claim of faithful reproduction occurred.

## Reusable structure

The [sample configuration](https://github.com/Vaidehi99/MultiAgentPrivacy/blob/6e7af5d8a8b8a803d8d51cefc01831d2e0724b41/data/experiment_10_def_adv_both.json) provides agent identities, agent-local tables, table-level sensitivity attributes, benign and sensitive goals, composition steps, final-inference references, agent descriptions, plans, and a prose combination-sensitive rule. Sample 10 has four agents, two tables each, and three rows per table: 24 table rows. Its linked company, owner, city and financial records are a useful external relational-task anchor.

The [README](https://github.com/Vaidehi99/MultiAgentPrivacy/blob/6e7af5d8a8b8a803d8d51cefc01831d2e0724b41/README.md) describes several inference-time defenses and separate plan-execution/final-inference evaluation. This supports considering external compositional tasks and preserving privacy/utility distinctions. It does not supply ready-made exact CCE preferences, shared-role SFT/DPO data, or an independently validated long-horizon CI oracle.

## Limits that matter for adaptation

1. In the [history defense implementation](https://github.com/Vaidehi99/MultiAgentPrivacy/blob/6e7af5d8a8b8a803d8d51cefc01831d2e0724b41/scripts/qwen_gemini_map_cot_sensitive_set_hist.py#L281), lines 315–316 and 388–393 supply previous queries by the same requester across agents. This is not the present project's recipient-delivered knowledge ledger. Lines 952–956 flatten each agent's table rows into a context; the table-level sensitivity field is not separately retained by that function. The configuration's prose defense is supplied at lines 973 and 318. Responses and intermediate query results also enter mutable agent context (410, 442, 668), so the runtime has a different visibility/state contract from the frozen typed environment.

2. The [evaluation script](https://github.com/Vaidehi99/MultiAgentPrivacy/blob/6e7af5d8a8b8a803d8d51cefc01831d2e0724b41/scripts/eval_leakage_exec.py) sends plan/trajectory/reference prompts to Qwen3-32B and parses model decisions for final inference, plan adherence, and plan execution. Those judgments are not deterministic closure or exact continuation certificates. Replacing them with a typed oracle would be an adaptation requiring its own source-to-oracle audit.

3. Sample 10 needs semantic qualification before use as a complete task oracle. Its sensitive goal uses unspecified high-income and significant-debt conditions. The reference mentions John (age 55, Dallas/urban, income 2.5M, debt 1.2M), while the tables also contain James (age 60, Houston/urban, income 3.0M, debt 1.8M). Under monotone high-income/debt threshold interpretations that include John, James also qualifies. This is a reviewer inference from the sample, not evidence of a repository-wide defect. It demonstrates why a prose reference should not be silently promoted to an exhaustive exact answer set. Introducing precise thresholds or changing completeness criteria must be labeled and validated as an adaptation.

4. No LICENSE/COPYING file appears in the reviewed [pinned root listing](https://github.com/Vaidehi99/MultiAgentPrivacy/tree/6e7af5d8a8b8a803d8d51cefc01831d2e0724b41), and the reviewed README has no license statement. Record `license_not_found_in_reviewed_root_and_readme`; do not infer an open training/redistribution license or a legal prohibition from this limited observation.

5. Preserve the original scenario as the source group. Agent/table rows, benign-versus-sensitive goals, alternative prompts/defenses, renamed entities, and generated attack trajectories from that same configuration are related variants. Multiple numbered files should not be declared independent task templates until their provenance and normalized relational structures are compared.

This source can anchor a subsequent bounded adaptation while keeping the KCI research objective. At present it supports source/schema provenance only, not a new benchmark result, human-verified CI labels, or natural-language generalization.
