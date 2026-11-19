"""Read-only localization after independent factor experiment audit.
Explains which factor cells fail; is descriptive, not an intervention or circuit proof.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
BASE=HERE.parent;ROOT=BASE.parents[1]
COMP=ROOT/"artifacts"/"20261117_composition_repair"
def rows(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def check():
    assert read(HERE/"AUDIT.json")["audit"]=="PASS"
    dev=rows(HERE/"dev.jsonl")
    labels=rows(COMP/"labels.jsonl")
    bystate={r["state_id"]:r["case"] for r in labels}
    oldstates=rows(COMP/"states.jsonl")
    result={}
    for arm in ("initial","task_replay_control","relation_factor"):
        fold=HERE/"study"/arm
        task={x["id"]:x["estimate"] for x in rows(fold/"dev"/"task_outputs.jsonl")}
        rel={x["id"]:x["estimate"] for x in rows(fold/"dev"/"relation_outputs.jsonl")}
        conf=defaultdict(Counter)
        conditional=Counter()
        for r in dev:
            key=(r["role"],r["permission"],r["full_result_required"],r["relation_target"])
            stat=conf[key]
            stat["n"]+=1
            tc=task[r["id"]]==str(r["task_target"])
            rc=rel[r["id"]]==str(r["relation_target"])
            stat["task_correct"]+=tc;stat["relation_correct"]+=rc
            conditional["both_correct"]+=tc and rc
            conditional["relation_correct_task_wrong"]+=rc and not tc
            conditional["relation_wrong_task_correct"]+=not rc and tc
            conditional["both_wrong"]+=not rc and not tc
        # On the native panel, classify unsafe choices and relation/task errors.
        relnative={x["id"]:x["estimate"] for x in rows(fold/"self_fact"/"relation_outputs.jsonl")}
        tasknative={x["id"]:x["estimate"] for x in rows(fold/"self_fact"/"direct_queries.jsonl")}
        choices=rows(fold/"self_fact"/"direct_actions.jsonl")
        native_questions=rows(COMP/"queries.jsonl")
        qid={(r["state_id"],r["action"]):r["id"] for r in native_questions}
        safety=Counter();examples=[]
        for state,choice in zip(oldstates,choices):
            assert state["id"]==choice["id"]
            selected=choice["selected"]
            if selected is None:
                safety["invalid"]+=1;continue
            case=bystate[state["id"]]
            candidate=next(x for x in case["candidates"] if x["action"]==selected)
            if candidate["immediate_safe"]:continue
            safety["unsafe"]+=1
            qkey=qid[(state["id"],selected)]
            fact_l=next(x for x in labels if x["id"]==qkey)
            original_fact=fact_l["relation_fact"]
            relation_ok=relnative[qkey]==str(original_fact)
            completion_ok=tasknative[qkey]==str(candidate["safe_completion"])
            safety["unsafe_with_wrong_relation_estimate"]+=not relation_ok
            safety["unsafe_with_wrong_task_estimate"]+=not completion_ok
            examples.append({"state_id":state["id"],"role":case["observation"]["role"],
                             "selected_action":selected,"actual_safe_completion":candidate["safe_completion"],
                             "pred_relation":relnative[qkey],"gold_relation":original_fact,
                             "pred_task":tasknative[qkey]})
        result[arm]={"dev_cells":{str(k):dict(v) for k,v in sorted(conf.items())},
                     "dev_factor_task_disagreement":dict(conditional),
                     "native_self_unsafe_factors":dict(safety),
                     "native_self_unsafe_examples":examples}
    (HERE/"ERROR_LOCALIZATION.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"localization":"COMPLETE","summary":{k:{
        "dev":v["dev_factor_task_disagreement"],
        "native_self_unsafe":v["native_self_unsafe_factors"]} for k,v in result.items()}},ensure_ascii=False))
if __name__=="__main__":check()
