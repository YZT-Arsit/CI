"""Summarize which K8 terminal directions depend on seed subset."""
from pathlib import Path
import json,collections
p=Path(r"D:\research\longhistory\artifacts\20261123_cce_phase2_mechanism_v1\TERMINAL_PROPAGATION_STABILITY.json")
x=json.loads(p.read_text(encoding="utf8"))
all=x["cells"]
print("K8 accepted unstable jack",sum(r["terminal_all_K8"] is not None and not r["jackknife_preserves_full"] for r in all),"of",sum(r["terminal_all_K8"] is not None for r in all))
print("K8 accepted unstable half",sum(r["terminal_all_K8"] is not None and not r["half_splits_preserve_full"] for r in all))
for r in all:
 print(r["cell_id"],"terminal",r["terminal_all_K8"],"leave1",list(set(str(x) for x in r["terminal_jackknife_K7"])),"twohalves",r["terminal_two_predefined_K4_halves"],"ci_abstain",r["CCE_abstain_reason"])
