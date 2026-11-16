# Bounded learned-relation mechanism diagnostic

This is an auxiliary CPU diagnostic, not a replacement for shared LLM policy
training or a claim of repaired task-level generalization.

Use a small trainable message-passing network. Its inputs are entity equality,
typed adjacency of the three DECLARED relation tables, an observed sensitive
endpoint marker, and the visible permission bit. No computed closure, risk bit,
future value, or target label is an input feature. Identifiers are discarded
after exact structured entity resolution: name invariance is architectural and
must not be claimed as a learned language capability.

Train on 256 synthetic graphs with two aliases and two case/owner keys. Each has
the same edge/node counts across opposite labels; key matching and permission
are independently balanced. Evaluate on separately generated graphs: fresh
bindings; five aliases/keys; and five aliases/keys with branching middle links.
The latter is a graph-composition stress test, not an independent source task or
a native DynamicTask replay. Exact relational labels are generated separately.

Predeclare 4 controls x 3 seeds (17,29,41): depth0, depth2, depth3, and depth3 with
shuffled training labels. Fixed 300 Adam steps, batch32, width8, learning rate.02.
No tuning on evaluation or selective reruns. Each worker <=9.5 seconds. Save all
weights, predictions, input hashes, finite-difference gradient and node-permutation
checks. Necessary diagnostic gate: depth3 >=95% accuracy in EVERY match/permission
stratum on EVERY held-out graph condition for EACH seed. This is not sufficient
for GPU GO: task policy learning, source diversity and business-cost validation
remain separate requirements even if this graph-only gate passes.
