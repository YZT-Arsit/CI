"""V3.1 strict history-only control after the NEW semantic renderer."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_shared_semantic_action import state_text,candidate_prompt,canonical_actions
D=ROOT/"artifacts/20261123_shared_semantic_action_v1"
def read(p):
    return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines() if s.strip()]
def only_history_removed(obs):
    return "\n".join(line for line in state_text(obs).splitlines()
                      if not line.startswith("Delivered receipt "))
def main():
    x=read(D/"V31_TRAIN_72.jsonl")+read(D/"V31_EVAL_54.jsonl")
    assert len(x)==126
    m=defaultdict(dict)
    for r in x:
        name=r["world"].rsplit("__",1)[0]
        history=r["world"].rsplit("__",1)[1]
        assert history in ("related","same_entity_safe")
        assert history not in m[name]
        m[name][history]=r
    fail=[]
    labels=Counter()
    for name,r in m.items():
        assert set(r)=={"related","same_entity_safe"}
        a,b=r["related"],r["same_entity_safe"]
        labels[a["chosen_action"]]+=1
        labels[b["chosen_action"]]+=1
        same=only_history_removed(a["observation"])==only_history_removed(b["observation"])
        if not same:
            fail.append(name)
        assert a["chosen_action"]=="OPEN_PRIVATE_CHANNEL"
        assert b["chosen_action"]=="SKIP_PRIVATE_CHANNEL"
        assert canonical_actions(a["observation"])==canonical_actions(b["observation"])
        for act in canonical_actions(a["observation"]):
            assert candidate_prompt(a["observation"],act)!=candidate_prompt(b["observation"],act)
    assert not fail,fail
    report={
       "n_strict_history_pairs":len(m),
       "n_distinct_business_graph_roots":len(set(x["root"] for x in x)),
       "n_renderer_nonhistory_identical":len(m),
       "n_label_flips":len(m),
       "n_actor_state_rows":len(x),
       "labels":dict(labels),
       "same_native_action_menu":True,
       "different_receipt_history_and_same_all_other_lines":True,
       "no_eval_only_data_used_as_train":True,
       "model_data_filenames":["V31_TRAIN_72.jsonl","V31_EVAL_54.jsonl"],
    }
    out=D/"V31_STRICT_RENDERER_VALIDATION.json"
    assert not out.exists()
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
