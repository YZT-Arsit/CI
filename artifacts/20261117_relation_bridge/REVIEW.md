# Independent read-only review

The existing rehearsal_review reviewer independently recounted all 3,264 responses and verified 27 sealed output files, all query/action/safety totals and the three seed gates. It recomputed all 24 history-family summaries. No counting, scoring or seal blocker was found. No model calls or source edits were made by the reviewer.

The audit repair was checked as exactly one replacement: JSON roundtrip on regenerated build_records output. Only labels had native tuple versus stored list differences; after normalization all matched. The original auditor, original failure log and repair hashes were verified. Labels, scores and gates were unchanged.

The review corrected an intermediate metric description: invariant both-correct counts are 42/116, 53/116 and 4/116; invariant prediction changes are 69/116, 55/116 and 110/116. The report uses these separate definitions. Truth has lower query and optimal-action accuracy than direct, while unsafe actions change from 3/1/5 to 0/1/1; the report does not characterize all metrics as worsening.

The reviewer agrees that the composition interface remains unreliable and that neither learned relation-source failure nor reuse is established. It recommends bounded matched query-supervision repair covering Stop, Minimal, permission and optional requirements, preserving shared-parameter action replay and direct-query retention. Sensitive contradictory notes must not become true-world supervision. Existing worlds are development diagnostics, not a fresh blind test. Final checkpoints and a separate base-world-held-out evaluation must be declared before further calls.

Archive restoration and final Git package verification are separate delivery checks, outside this review. Large GPU remains NO_GO; the full goal remains incomplete.
