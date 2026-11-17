# Independent agent review

Prelaunch review reconstructed all18role/depth/taskcells,144scheduleentries with every batch repeated8times,336prompts and192changing-query rows. It verified outcome-independent selection, identical forward code and teacher actor inputs, and query-only backward separation from actor_loss. Frozen hashes, checkpoint/dependency checks, step accounting, identity receipts and byte-reused oracle checks passed.

Two minor reporting issues were corrected: the generation limit is max_time8 with a9.5second validity cutoff, not a hard watchdog; fixed-label query correctness and invalid counts are now reported separately from changing-label queries. Earlier files remain in prelaunch_review. Independent re-review verified that only audit.py and PLAN.md hashes changed; inputs, schedule, checkpoint, dependencies and training computation were identical.

The reviewer did not run models. A Windows tokenizer-only attempt was unavailable because the pinned dependencies reside in WSL; the coordinator separately verified WSL worker import and4CPUthreads. Runtime token counts are retained for all evaluated query prompts. Review is agent code/evidence review, not human validation of natural norms.

Completed-output review independently rescored all summaries: initial12/11, query-only37/21, joint39/20 correct changing pairs out of48 for original/renamed names. All models retained144/144fixed-label queries, equal-token controls and both oracle gates with zero unsafe/invalid actions. Neither trained arm passed complete query fitting. These results do not establish actor loss as the primary failure cause.

The separate analytic bit-weight audit was reviewed against all24records and the actual CE source. It was clarified to concern conditional0/1logodds at fixed binary probability mass, not arbitrary full-vocabulary logit gradients. Original wording is retained in bit_review_revision. No empirical mechanism claim follows solely from that algebra.
