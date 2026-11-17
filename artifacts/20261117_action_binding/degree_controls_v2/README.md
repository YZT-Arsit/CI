# Anchored connectivity controls: qualified scope

Independent review verified all 16 pairs/32 worlds: same histories, tickets, nodes, per-node in/out degrees, edge counts and component sizes. Every provider has two equal directed cycles. Neither direct ticket-to-history nor history-to-ticket edges exist. Unique safe choices flip; reachable anchor distances are 2/3/4/6.

Model inputs must use the existing `kci.relation_language.prompt` renderer, which shuffles provider edges, facts and options and renames entities. The raw construction stores the ticket's cycle first; exposing that unshuffled order would leak the label through first-half membership. World IDs, side IDs and label metadata are also excluded from model input. Do not train a raw sequence encoder on the saved ordering and call the result a relation-learning test.

This family tests anchored component membership. Directed cycles make reachability symmetric within each component, so undirected connectivity is enough. It does not by itself establish general direction-sensitive reachability; the separate internal-edge-reversal family tests direction, with its own local-degree limitations. No finite diagnostic suite proves the absence of every possible shortcut.

The earlier `../degree_controls/` family retains a perfect one-edge closing-edge cue and remains a documented failed diagnostic design. These corrected controls have not yet established any model success.
