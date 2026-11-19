"""Model-tokenizer-only preflight; no updates, no paid API.
Checks frozen data, supervised EOS, padded width and model/source binding.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json,sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent;ROOT=BASE.parents[1]
sys.path[:0]=[str(BASE),str(ROOT/"artifacts"/"20261116_relation_solution")]
from lm_common import AutoTokenizer,env
import build_signals
def sha(p):return sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return build_signals.load_rows(p)
def encode(tok,text,answer):
    inp=tok.apply_chat_template([{"role":"user","content":text}],tokenize=True,add_generation_prompt=True)
    suf=tok.encode(answer+tok.eos_token,add_special_tokens=False)
    assert suf[-1]==tok.eos_token_id and tok.decode([suf[-2]])==answer[-1]
    assert len(inp)+len(suf)<1024
    return len(inp)+len(suf)
def main():
    plan=read(HERE/"PLAN.json")
    if (HERE/"TOKEN_PREFLIGHT.json").exists():raise FileExistsError("No overwrite")
    for name,h in plan["data_sha256"].items():assert sha(HERE/(name+".jsonl"))==h
    oldactor=ROOT/"artifacts"/"20261117_composition_repair"/"training_17.jsonl"
    assert sha(oldactor)==plan["source_sha256"]["old_actor_replay"]
    cp=ROOT/plan["start_checkpoint"]
    assert read(cp/"CHECKPOINT_SEALED.json")==plan["start_seal"]
    assert all(sha(cp/"adapter"/n)==h for n,h in plan["start_seal"]["adapter_hashes"].items())
    tok=AutoTokenizer.from_pretrained(env["native_lm_path"],local_files_only=True)
    train={r["id"]:r for r in rows(HERE/"train.jsonl")}
    schedule=rows(HERE/"schedule.jsonl")
    old=rows(oldactor)
    assert len(train)==128 and len(schedule)==64
    metrics={}
    for arm in plan["arms"]:
        task_padded=extra_padded=actor_padded=0
        max_width=0
        allids=[]
        for item in schedule:
            chosen=[train[i] for i in item["query_ids"]]
            global_l=[encode(tok,r["task_query"],"ANSWER:\n"+str(r["task_target"])) for r in chosen]
            extras=[encode(tok,r["task_query"] if arm=="task_replay_control" else r["relation_query"],
                           "ANSWER:\n"+str(r["task_target"] if arm=="task_replay_control" else r["relation_target"]))
                    for r in chosen]
            actor=old[item["actor_source_step"]]["actor"]
            al=[encode(tok,a["prompt"],a["answer"]) for a in actor]
            task_padded+=max(global_l)*4;extra_padded+=max(extras)*4;actor_padded+=max(al)*4
            max_width=max(max_width,*global_l,*extras,*al)
            allids.extend(item["query_ids"])
        assert Counter(allids)==Counter({k:2 for k in train})
        metrics[arm]={"task_padded_tokens":task_padded,"extra_padded_tokens":extra_padded,
                      "actor_padded_tokens":actor_padded,
                      "max_sequence_length":max_width,
                      "train_task_labels":dict(Counter(r["task_target"] for r in train.values())),
                      "train_relation_labels":dict(Counter(r["relation_target"] for r in train.values()))}
    for name in ("dev","confirm_depth4"):
        panel=rows(HERE/(name+".jsonl"))
        for r in panel:
            encode(tok,r["task_query"],"ANSWER:\n"+str(r["task_target"]))
            encode(tok,r["relation_query"],"ANSWER:\n"+str(r["relation_target"]))
    assert metrics["task_replay_control"]["task_padded_tokens"]==metrics["relation_factor"]["task_padded_tokens"]
    assert metrics["task_replay_control"]["actor_padded_tokens"]==metrics["relation_factor"]["actor_padded_tokens"]
    report={"preflight":"PASS","model_calls":0,"plan_sha256":sha(HERE/"PLAN.json"),
            "metrics":metrics,
            "comparison_limit":"Equal query/actor counts and updates, but extra objective and padded tokens differ",
            "train_ready_for_formal_claim":False}
    (HERE/"TOKEN_PREFLIGHT.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))
if __name__=="__main__":main()
