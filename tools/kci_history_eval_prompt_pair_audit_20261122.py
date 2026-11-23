"""Native message-history counterfactual inspection, without changing any data."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
f=ROOT/"artifacts"/"20261122_history_evaluator_dev_v1"/"PAIR_STATES.jsonl"
r=[json.loads(x) for x in f.read_text(encoding="utf8").splitlines()]
for root in ("DG01","DG05","DG18"):
    matched=next(x["world"] for x in r if x["root"]==root and x["source_mode"]=="matched")
    rows={x["branch"]:x for x in r if x["world"]==matched}
    print("ROOT",root,"WORLD",matched)
    for branch,x in rows.items():
        prompt=x["visible_prompt"]
        receipt=[line for line in prompt.splitlines() if line.startswith("Delivered receipt")]
        rules=[line for line in prompt.splitlines() if line.startswith("Inference rule")]
        print("BRANCH",branch,"label",x["aware_label"],"blind",x["blind_label"],
              "receipt_count",len(receipt),"rule_count",len(rules))
        print("RECEIPTS",*receipt,sep="\n  ")
        if branch=="risky":
            print("RULES",*rules,sep="\n  ")
    print("DIFFERENT_RECEIPT_LIST",rows["risky"]["recipient_receipts"]!=rows["safe"]["recipient_receipts"])
