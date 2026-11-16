# Three-hour relation and safety/utility repair

Start 2026-11-16 06:34 UTC. Target finish 09:34 UTC. Main only, baseline
1a03503ae2734210e385bda98ef883149332d22d. CPU only; experiments in <=9.5s
chunks, setup/download separately measured. Prior evidence remains immutable.

Goal: shared learned action policy responds to trusted history/task dependence,
generalizes across symbols and relational compositions, and preserves BOTH safety
and achievable task success. The original language-model research goal remains.

Hypotheses to test, not assume:
1. Pair-only supervision drops state/action coverage and conflates safety with
   attainable task value. Dense executed action vectors and separate prediction
   targets reduce refusal collapse. Ties remain ties, not arbitrary labels.
2. Repeated unconstrained tanh propagation can forget/reverse evidence at unseen
   depths. A learned local relation operator with shared, monotone composition
   may generalize better. Architectural bias must be disclosed, not called
   learned free-text inference or a novel method.
3. Training-only local proof/binding supervision and symbol interventions may
   help a language model. Evaluate a genuine small pretrained CPU model if its
   bounded step time is feasible; finite synthetic learners alone cannot close
   the original LLM goal.

Data: retain external pinned task computations; construct paired dynamic CI
histories and norm/obligation controls with exact executed labels. Use new seeds
and paired groups, train depths1..3, dev1..4, blind tests4/6/10 and larger noise,
wrong direction/broken link/branch/cycle controls. Never split renamed variants
across groups. Both/no feasible and legal-combination cases are mandatory. Add
preselected PiSAs Meeting10..17 at the SAME pinned revision; report all failures
or unsupported schemas. Their CI histories remain synthetic.

Method comparisons: old V5 frozen policy; data-only correction under an old
architecture; decomposed learned safety/value policy; relation-structure and
label-shuffle controls. Any custom monotone core remains an adapted structured
model. No runtime exact closure, oracle mask, fallback or rule-overridden action
may count as learned unassisted success. Training-only targets may use verified
enumeration/closure, with labels separate from model inputs.

Promotion: each declared viable stratum >=95% safe-success for each of three
seeds, paired two-sided success >=95%, no-feasible cases >=95% CI. Report every
seed, error and refusal; do not pool repeated episodes as independent samples.
For an LLM pilot additionally disclose format/candidate-scoring constraints and
native-vs-adapted scope. Full LLM/GPU readiness is separate from a structured
mechanism gate. No more tuning on the old test set; new main test predictions
remain sealed until chosen configurations/checkpoints are frozen.

Development uses fresh train/dev data. Record each pilot and reason for changes.
Freeze final exact configs before main evaluation. Report limitations, sources,
dataset counts, tests, outcomes and readiness; push the reviewable snapshot to
main. Follow the scoped ECC mle-workflow data-contract/error-analysis lanes;
do not introduce unrelated MLOps machinery.
