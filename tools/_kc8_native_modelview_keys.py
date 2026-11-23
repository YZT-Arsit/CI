from pathlib import Path
import sys
ROOT=Path(r"D:\research\longhistory")
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_kc8_real4b_train_diagnostic import load_native_snapshot,clone_prefix
cs,E,M,dig,verify=load_native_snapshot()
c=next(c for c in cs() if c.id=="KC801")
e=E(c,"H1")
s=clone_prefix(e)
v=M(e.actor_observation(s))
print("model view keys",list(v))
print("extra",set(v)-{"actor","stage","model_prompt","prompt_sha256","available_actions","complete_role_observable_empirical_probability","no_ground_truth_label_or_specific_future_realization","no_hidden_teacher_or_future_truth","role_observation_sha256","actual_public_routing_and_automatic_source_contract_present","actual_signed_tool_status_is_visible"})
