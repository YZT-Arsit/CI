# Additional read-only diagnostics

Declared while the original worker is running, before scoring this study. `diagnostics.py` will decompose saved outputs into binary confusion, valid witnesses, false `none`, copied-but-invalid names, names absent from the input, and format/time failures. It will show family, noise, vocabulary and valid-set-size counts. Cross-tabulation of the two separate requests is descriptive association; a valid generated witness does not establish the internal cause of a binary decision.

`test_scores.py` uses fabricated responses to check perfect, constant, multiple-answer, time-cutoff, identity and length-control cases. These tests perform no model calls, add no training, and change no frozen source, scoring rule or acceptance threshold. The original 1,152-update, 9,504-new-response plan remains unchanged. New scripts are not part of the original frozen worker dependencies and will be hashed in their own receipts and delivery.
