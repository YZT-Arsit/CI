# Reproduce the external norm probe

Restore this archive and the previous fixed-state/external-norm archive. Upstream CAMBER CSV, README and LICENSE are pinned by Git commit, Git blob hash and SHA256; restoration must recover the original bytes. `fetch.py` refuses an existing source manifest. `build.py` refuses an existing PLAN and selects before model output; it records one exact-source mismatch excluded before sampling.

`run.py` uses only PLAN and observations, never labels. It verifies the local Ollama model digest, requires no initially loaded models, records source hashes, and sends64independent single-user-message requests with fixed CPU options. There is no conversational history across cases, no retry or answer-dependent prompt change. A completed request over25seconds is retained separately and stops the run; the client timeout itself is not a hard server deadline. Free responses are sealed before `audit.py` reads labels.

```powershell
python -X utf8 -B artifacts/20261117_external_norm_probe/audit.py
python -X utf8 -B artifacts/20261117_external_norm_probe/position_followup/audit.py
```

The auditor verifies source hashes, exact adapted prompt construction from the five allowed fields, label/metadata separation, class balance, option order, recorded response identity and wall-time acceptance. It reports label agreement and abstention separately. It does not execute the original CAMBER evaluation suite or a business workflow.

The primary scorer is immutable after inference and both preregistered parsers require an ANSWER prefix. Its64bare-Cresponses remain invalid. The separately declared position_followup changes only option ordering, accepts bare letters under its own frozen scoring plan, and hashes all parent inputs, labels and outputs. It adds64requests but zero independent cases. Do not replace primary scores with follow-up scores or relabel primary formatting failures as64norm errors.

For a new model run, use a fresh copy/output directory, reverify model availability and digest, and preserve the original evidence. Do not run concurrently with CPU training. `RELEASE_CHECK.json` records whether the probe's model is absent from Ollama after its unload request; an unload response without this check is not resource-release evidence.
