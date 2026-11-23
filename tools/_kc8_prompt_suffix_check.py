from pathlib import Path
import json
p=Path(r"D:\research\kci_ci_causal_coverage_v2_20261122\artifacts\20261122_kci_causal_credit_method_handoff_v1")
for name in ("MODEL_ONLY_TRAIN_A.jsonl","MODEL_ONLY_TRAIN_DOWNSTREAM.jsonl"):
 rows=[json.loads(x) for x in (p/name).open(encoding="utf8")]
 for ix in (0,min(1,len(rows)-1)):
  r=rows[ix]
  print(name,ix,"END_SNIPPET",repr(r['model_prompt'][-900:]),flush=True)
