# Strict CCE with a verified executed-reference contract

## What cannot be inferred

For any finite n and an observed zero-violation sample, a reference with violation
probability epsilon > 0 can produce that sample with probability (1-epsilon)^n.
Thus sampled zeros alone do not prove exactly equal CI risks. A small confidence
upper bound is not exact equality and cannot justify the next lexicographic tier.
The raw/unrestricted path in stream_cce.py retains UNKNOWN for such lower tiers.

## Implemented alternative

The reference being evaluated is explicitly:

`frozen LLM proposal -> typed canonical decoder -> verified execution gateway -> tool`.

Raw drafts, rejected drafts, fallback actions, and executed actions are recorded
separately. The gateway is a reference-policy contract, not a claim that the raw
LLM is safe or that the target model has learned the constraint.

The gateway proves properties over the finite task state/action transition graph,
not the probability distribution of all possible LLM strings. It accepts a raw
proposal only if its decoded action belongs to the verified current-state set.
Malformed, unsafe or unsupported drafts cannot directly invoke a tool. Model
timeouts/errors are recorded and trigger the same fixed fallback. Unmodelled
external side effects are out of scope, never silently certified.

CI-only mode filters to verified safe transitions. If no safe action exists it
halts without delivering anything. Every admitted transition preserves CI, so
executed CI probability is 1 for ANY distribution over raw proposals. Success may
still vary, and can be compared by simultaneous confidence intervals. Equal
success is not inferred from finite samples.

Safe-success mode additionally keeps only actions with a verified safe completion
inside the remaining finite horizon. Initially both compared actions must already
be safe and viable: an infeasible/unsafe first action cannot be replaced to fake
its value. At every active viable state an eligible action exists. Each executed
action decreases the remaining horizon. Induction therefore reaches a successful
safe terminal state. This establishes equal CI and equal success for every raw
proposal distribution, allowing statistical comparison of executed completion
cost. This viability proof is exact finite planning and may be expensive or absent
in general workflows; without it, the cost tier remains UNKNOWN.

## Binding and limits

- Bind actual source data, source/mailbox ACLs, implementation hashes, starting
  state, compared first actions, raw reference identity, gateway mode and cost unit.
- Verify the transition graph has not changed. Every sample contains registered
  executed traces bound to its seed; raw Trial tuples cannot borrow the certificate.
- Task success consumes actual message evidence; oracle/private truth remains in
  the verifier. Source ACL revocation, missing inputs and drift revoke certification.
- Confidence intervals still require fixed protocol and appropriate independent
  samples from the frozen executed reference. The structural invariant does not
  establish IID, representative sampling, statistical power or method efficacy.
- Costs can be canonical payload UTF-8 bytes actually delivered, or explicitly
  synthetic API steps. Neither is presented as measured physical latency.
- For cost intervals, the verified graph supplies each branch's minimum/maximum
  terminal cost BEFORE sampling. A paired difference lies in
  [min(B)-max(A), max(B)-min(A)]. Hoeffding uses this certified range width,
  not sample extrema. This improves power without claiming risk equality from
  data or enumerating raw LLM outputs. Broad-range pilot UNKNOWNs are retained.
- A successful fallback is not a successful raw-LLM decision. Report intervention
  rates and unassisted decisions, and never use gateway success as learning evidence.
- The current proof supports two finite canonical tool contracts with synthetic
  CI norms. It does not solve arbitrary free-text CI entailment or unrestricted
  real-world LLM execution. Those still require a sound verifier/adapter; UNKNOWN
  is the correct result where such a contract is unavailable.

This preserves strict CI > compliant success > cost, rather than introducing an
epsilon-risk tradeoff. It makes a precisely qualified executed reference usable
without requiring a full-support proof about the raw LLM itself.
