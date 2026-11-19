"""Tokenizer and matched-budget preflight. Does not touch model weights."""
import hashlib,json,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1];COMP=ROOT/"artifacts"/"20261117_composition_repair"
sys.path.insert(0,str(ROOT/"artifacts"/"20261116_relation_solution"))
from lm_common import AutoTokenizer,env
import build_signals
S=HERE/"cpu_matched_01"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def tok_ids(tok,prompt,answer):
    context=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True)
    tail=tok.encode(answer+tok.eos_token,add_special_tokens=False)
    assert len(tail)>=2 and tail[-1]==tok.eos_token_id and tok.decode([tail[-2]])==answer[-1]
    return len(context)+len(tail),len(context)
def main():
    plan=read(S/"PLAN.json");assert plan["updates_per_arm"]==32
    for name,expected in plan["frozen_data_hashes"].items():assert sha(S/name)==expected
    tok=AutoTokenizer.from_pretrained(env["native_lm_path"],local_files_only=True)
    lookup={r["id"]:r for r in build_signals.load_rows(S/"train_panel.jsonl")}
    all_old=build_signals.load_rows(COMP/"training_17.jsonl")
    metrics={}
    for arm in plan["arms"]:
        steps=build_signals.load_rows(S/(arm+"_steps.jsonl"))
        qsize=[];asize=[];cases=[]
        for step,rec in enumerate(steps):
            assert rec["step"]==step and rec["actor_source_step"]==step and len(rec["query_ids"])==4
            items=[lookup[i] for i in rec["query_ids"]]
            cases.extend(rec["query_ids"])
            q=[tok_ids(tok,e["prompt"],"ANSWER:\n"+str(e["label"])) for e in items]
            a=[tok_ids(tok,x["prompt"],x["answer"]) for x in all_old[step]["actor"]]
            assert all(l<1024 for l,_ in q+a)
            qsize.append(max(l for l,_ in q)*4)
            asize.append(max(l for l,_ in a)*4)
        assert len(cases)==128 and len(set(cases))==128
        metrics[arm]={"total_padded_query_tokens":sum(qsize),
                      "total_padded_actor_tokens":sum(asize),
                      "max_padded_query_tokens_per_update":max(qsize),
                      "train_label_distribution":dict(Counter(lookup[x]["label"] for x in cases))}
    assert metrics["paired"]["total_padded_actor_tokens"]==metrics["shuffled"]["total_padded_actor_tokens"]
    assert metrics["paired"]["train_label_distribution"]==metrics["shuffled"]["train_label_distribution"]
    dev=build_signals.load_rows(S/"dev_panel.jsonl")
    lengths=[tok_ids(tok,x["prompt"],"ANSWER:\n"+str(x["label"]))[0] for x in dev]
    assert len(dev)==128 and max(lengths)<1024
    out={"model_calls":0,"preflight":"PASS","total_dev_rows":len(dev),"max_dev_tokens":max(lengths),
         "training":metrics,"plan_hash":sha(S/"PLAN.json"),
         "metric_note":"Padded tokens differ across batch grouping; equal optimizer updates and examples do NOT imply equal FLOPs."}
    (S/"TOKEN_PREFLIGHT.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out))
if __name__=="__main__":main()
