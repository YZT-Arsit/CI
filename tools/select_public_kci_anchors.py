import json,ast,re
from collections import defaultdict
from pathlib import Path
q=Path(r"D:\research\kci_public_refs_20261119\tau2-bench\data\tau2\domains\retail\tasks.json")
z=json.loads(q.read_text(encoding="utf-8"))
types=defaultdict(list)
for t in z:
    tools=tuple(dict.fromkeys(x['name'] for x in t.get('evaluation_criteria',{}).get('actions',[])))
    if tools:
        for tool in tools:
            if tool in ("modify_pending_order_address","modify_user_address","cancel_pending_order",
              "exchange_delivered_order_items","return_delivered_order_items","modify_pending_order_items"):
                types[tool].append((t["id"],t.get("user_scenario",{}).get("instructions",{}).get("reason_for_call","")[:135],tools))
for k,items in types.items():
    print("\nTOOL",k,"COUNT",len(items))
    for x in items[:6]:print(x)
