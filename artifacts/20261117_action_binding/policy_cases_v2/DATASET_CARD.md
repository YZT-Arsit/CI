# Source-grounded clause DEV cases

28 authored hypothetical cases, 14 condition groups, four task descriptions, two clauses from the pinned official GitLab handbook. These are clause-comprehension inputs; no real customer data, executable business task, or universal privacy norm is claimed.

The first 24 cases received source-based review with expected labels withheld. Agreement was 22/24: two apparent exception-positive cases omitted the required ZenDesk logging destination. Those inputs are preserved and labeled UNKNOWN. Four additions explicitly distinguish ZenDesk from personal notes. The same reviewer independently judged the additions with their expected labels withheld, agreeing on all four. This is model review, not human review, and does not remove shared-model/source-interpretation risks.

PASS means only that the named clause requirements or exception eligibility are satisfied under the stated hypothetical facts. It does not mean the whole workflow is compliant. Future documentation/deletion obligations are promised conditions; their actual completion has not been executed or verified. Missing required facts stay UNKNOWN. Known disqualifying facts can justify FAIL even if another fact is missing.

All cases are DEV. Keep paired task wordings of the same condition together in future random splits. Rewording the task does not create an independent policy source or real task instance. These cases are not eligible for action-policy training or for claiming source-level held-out generalization. The original provisional labels and the two disagreements remain in `../policy_cases/`.

For any later model probe, expose only the scoped question, hypothetical facts, task description and vetted source context. IDs, condition-group names, expected labels, review judgments and qualification metadata must not enter model inputs.

Source: official GitLab handbook revision `e1c706abbe174add18ac09a72976df2a79d0178c`; exact source URLs and SHA256 hashes are included per case. The source organization is GitLab. These clauses are not silently applied to the fictional telecom organization in the separate tau2 replay.
