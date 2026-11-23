"""List actual CCE K8 per-cell evidence without altering any result."""
from pathlib import Path
import json
root=Path(r"D:\research\longhistory\artifacts\20261123_cce_phase2_mechanism_v1\CCE_ANALYSIS_K8")
rows=[json.loads(s) for s in (root/"CCE_PAIR_PREFERENCES.jsonl").read_text(encoding="utf8").splitlines()]
for x in rows:
 b=x["CCE_sample_branch_stats"]
 a=x["A_knowledge_delta"]
 q=x["Qstar_late_join_unused_for_labeling"]
 choice=x["simple_terminal_propagation"]["chosen_if_any"]
 print(x["cell_id"],x["deadline_class"],
       "A",b["A"]["unsafe_violation"],b["A"]["ci_safe_success"],b["A"]["safe_abort"],b["A"]["deadline_missed"],
       "B",b["B"]["unsafe_violation"],b["B"]["ci_safe_success"],b["B"]["safe_abort"],b["B"]["deadline_missed"],
       "CCE",x["CCE_preference"] or "ABSTAIN "+x["CCE_abstain_reason"],
       "TERM",choice or "tie",
       "Q*",q.get("Qstar_preferred_A_if_defined"),
       "delta",x["simple_terminal_propagation"]["difference_A_minus_B"])
