# Review history

The new one-round label function was test-driven from an explicitly unimplemented placeholder; `RED.log` is not presented as a pre-existing production bug. The exhaustive Boolean-matrix comparison and direction tests pass. A separate JSON-roundtrip regression caught tuple/list reconstruction mismatch before inference; the old source/plan and failure log are retained under `prelaunch_json_revision`.

Independent review verified334dependency hashes, perfect-output scoring and source-label separation. It identified that noise-zero groups and controls repeat structural templates; the protocol now states this explicitly. It also identified the lack of a separately balanced direction control. Before model calls, the prior plan/data/sources were retained under `prelaunch_direction_revision`, and half the pair groups were replaced with reversed-cycle pairs that preserve the undirected edge multiset and all node degrees. The total response count stayed1280. A valid failing direction test and the passing four-test suite are preserved.

Revised review verified16cells×8pairs per mode,128pairs and64controls; a deliberately incorrect answer fails the necessary gate. Tokenizer preflight subsequently verified256 paired own/padded lengths and the181/631token maxima against the1024limit. Frozen worker execution started only after those checks. No reviewer performed inference or edited files.

Conditional local training design was written while frozen probing was in progress and before results were scored. The latest checkpoint failed both modes, activating that separately registered follow-up. Neither this probe's labels nor its failed cases were substituted into the confirmation set. Structural templates and vocabulary reuse remain explicit.
