# Attribution and scope

Data: CAMBER PrivacyLens+, from the Google Contextual Privacy and Security group, associated with [Privacy Reasoning in Ambiguous Contexts](https://arxiv.org/abs/2506.12241). Author repository pinned to [62e5d4529d291ff0eb1f8f9565f035e8a8827589](https://github.com/google-parfait/contextual-privacy-and-security/tree/62e5d4529d291ff0eb1f8f9565f035e8a8827589). The upstream README and full CC-BY4.0 LICENSE are retained unmodified under `source/`; all downloaded files match published Git blob hashes.

CAMBER derives these records from [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens). The original PrivacyLens copy, its attribution, license and pinned source manifest remain in `../20261117_external_norms/`. This run adds only local sampling, neutral option-order prompts, separated evaluation labels and model-output audits. Upstream CSV bytes are unchanged. New prompts are adapted evaluation inputs, not an official benchmark implementation. No dataset record is added to model training.

The `no_expansion` subset still includes synthetic positive seed modifications. Excluding expanded narratives does not make the positive/negative pairs identical apart from consent. A source-match audit conservatively excludes one negative seed differing only in a spelling correction (`editting` to `editing`); this is a byte-exact matching decision, not evidence of a norm disagreement.

The16groups are distinct normalized source titles, not independently verified source documents or32independent tasks. Benchmark-label agreement, abstention, and option-order robustness are the measured quantities. Real-world legal applicability, organization-specific exceptions, human validation by this project, task execution and task utility are not established.
