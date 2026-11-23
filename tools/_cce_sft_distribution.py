from pathlib import Path
from collections import Counter,defaultdict
import json
D=Path(r"D:\research\longhistory\artifacts\20261124_cce_downstream_sft_v1")
d=[json.loads(l) for l in (D/"SHARED_ROLE_ACTION_SFT_TRAIN_ONLY.jsonl").open(encoding="utf8")]
w=[json.loads(l) for l in (D/"NATIVE_SAFE_FULL_TEAM_WITNESSES_TRAIN_ONLY.jsonl").open(encoding="utf8")]
print("witness by",Counter((x["root"],x["variant"],x["authorized"]) for x in w))
print("SFT phase per root",Counter((x["root"],x["stage"]) for x in d))
print("VERIF examples")
for x in d:
 if x["chosen_action"] in ("VERIFY_PRIVATE_TARGET","OPEN_PRIVATE_CHANNEL","SEND_RAW_BROKER"):
  print(x["root"],x["stage"],x["chosen_action"],x["role"], x["multiple_successful_native_action_witnesses"])
print("tied choices",Counter((x["stage"],tuple(sorted(x["multiple_successful_native_action_witnesses"]))) for x in d if len(x["multiple_successful_native_action_witnesses"])>1))
