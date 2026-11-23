"""Freeze pure recipient-history-only matched counterfactuals for micro-fit test."""
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts"/"20261122_history_evaluator_dev_v1"
def load(f):return [json.loads(x) for x in f.read_text(encoding="utf8").splitlines() if x.strip()]
def main():
    raw=load(D/"PAIR_STATES.jsonl")
    pairs=defaultdict(dict)
    for r in raw:pairs[r["world"]][r["branch"]]=r
    teacher={}
    for mode in ("VISIBLE","BLIND"):
        a=load(D/("TRAIN_"+mode+".jsonl"))
        select=[x for x in a if x["root"] in ("DG01","DG05") and
                next(r for r in raw if r["world"]==x["world"])["source_mode"]=="matched"]
        assert len(select)==8
        assert len(set(x["world"] for x in select))==4
        assert Counter(x["branch"] for x in select)=={"risky":4,"safe":4}
        assert Counter(x["chosen_action"] for x in select)==(
          {"SEND_RAW_R":4,"VERIFY_PRIVATE_R":4} if mode=="VISIBLE" else {"SEND_RAW_R":8})
        for x in select:
            other=pairs[x["world"]]
            curr=other[x["branch"]]["visible_prompt"]
            def strip(p):return [r for r in p.splitlines() if not r.startswith("Delivered receipt ")]
            assert strip(other["risky"]["visible_prompt"])==strip(other["safe"]["visible_prompt"])
            assert curr==x["prompt"]
        path=D/("TRAIN_"+mode+"_STRICT8.jsonl")
        assert not path.exists()
        path.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in select),encoding="utf8")
        teacher[mode]=select
    for x,y in zip(teacher["VISIBLE"],teacher["BLIND"]):
        assert (x["prompt"],x["all_executable_actions"],x["state_hash"])==(
             y["prompt"],y["all_executable_actions"],y["state_hash"])
    meta={"n_states_per_arm":8,"n_paired_worlds":4,"n_independent_roots":2,
       "all_nonhistory_prompt_lines_identical_within_every_pair":True,
       "inputs_byte_identical_between_arms":True,
       "labels":{"visible":{"SEND_RAW_R":4,"VERIFY_PRIVATE_R":4},
                 "blind":{"SEND_RAW_R":8}},
       "files_sha256":{mode:sha256((D/("TRAIN_"+mode+"_STRICT8.jsonl")).read_bytes()).hexdigest()
                       for mode in ("VISIBLE","BLIND")}}
    dest=D/"STRICT8_MANIFEST.json"
    assert not dest.exists()
    dest.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
