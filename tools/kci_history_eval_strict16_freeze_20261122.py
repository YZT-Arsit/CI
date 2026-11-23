"""Strict balanced two-sided train-only diagnostic for the same evaluator cell.
Same actual 16 actor observations for both teacher arms; two independent roots.
"""
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"artifacts"/"20261122_history_evaluator_dev_v1"
KEEP={"DG01","DG05"}
def read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
def main():
    arms={}
    for teacher in ("VISIBLE","BLIND"):
        orig=read(DATA/("TRAIN_"+teacher+".jsonl"))
        assert len(orig)==32
        items=[x for x in orig if x["root"] in KEEP]
        assert len(items)==16 and {x["root"] for x in items}==KEEP
        assert len({x["world"] for x in items})==8
        assert Counter(x["branch"] for x in items)=={"risky":8,"safe":8}
        labels=Counter(x["chosen_action"] for x in items)
        assert labels==({"SEND_RAW_R":8,"VERIFY_PRIVATE_R":8} if teacher=="VISIBLE" else {"SEND_RAW_R":16})
        path=DATA/("TRAIN_"+teacher+"_STRICT16.jsonl")
        assert not path.exists()
        path.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in items),encoding="utf8")
        arms[teacher]=items
    for a,b in zip(arms["VISIBLE"],arms["BLIND"]):
        assert (a["prompt"],a["all_executable_actions"],a["observation_hash"],a["state_hash"])==(
            b["prompt"],b["all_executable_actions"],b["observation_hash"],b["state_hash"])
    m={
        "target":"ONLY history-aware vs history-blind teacher strict 2-sided 16-state train diagnostic",
        "n_state_rows_per_arm":16,"n_paired_worlds":8,"n_independent_train_roots":2,
        "roots":sorted(KEEP),"risk_safe_state_pairs_per_world":2,
        "all_actor_inputs_and_action_options_identical":True,
        "teacher_labels":{"visible":dict(Counter(x["chosen_action"] for x in arms["VISIBLE"])),
                          "blind":dict(Counter(x["chosen_action"] for x in arms["BLIND"]))},
        "source_splits":"train only; no DG18 or other dev/exploratory test world",
        "sha256":{key:sha256((DATA/("TRAIN_"+key+"_STRICT16.jsonl")).read_bytes()).hexdigest()
                  for key in ("VISIBLE","BLIND")},
    }
    f=DATA/"STRICT16_MANIFEST.json"
    assert not f.exists()
    f.write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(m,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
