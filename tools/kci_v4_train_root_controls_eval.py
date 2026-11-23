"""Frozen V4 TRAIN root controls: unseen hold/equivalence within seen roots.

Both original and entity-alias actor-only views. NO DG34, DG35 or DG36
models/labels are parsed. Control-only states are not used for fitting.
Strict score equality => UNRESOLVED, never correct by action ordering.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,time,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_v4_semantic_action import scoring,assert_views_position_invariant
from kci_v4_pairwise_strict_gate import exact_action,strict_correct
RAW="SEND_RAW_R";VERIFY="VERIFY_PRIVATE_R"
def jl(p):return [json.loads(x) for x in Path(p).read_text(encoding="utf8").splitlines() if x.strip()]
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def stats(xs):
    roots=defaultdict(dict)
    for x in xs:
        assert x["root"] in ("DG31","DG32","DG33")
        roots[x["root"]][x["condition"]]=x
    kinds={"Flip":("risky","safe"),
           "Hold":("safe","hold"),
           "Authorized":("risky","authorized"),
           "EquivalentHistoryOrder":("safe","equivalent_order")}
    return {
       "n_states":len(xs),"n_roots":len(roots),
       "strictly_correct":sum(x["strict"] for x in xs),
       "n_unresolved_ties":sum(x["prediction"]=="UNRESOLVED_TIE" for x in xs),
       "seen_train_states_correct":sum(x["strict"] for x in xs if x["seen_train"]),
       "seen_train_den":sum(x["seen_train"] for x in xs),
       "unseen_control_states_correct":sum(x["strict"] for x in xs if not x["seen_train"]),
       "unseen_control_den":sum(not x["seen_train"] for x in xs),
       "pairs":{name:{"correct_roots":sum(g[a]["strict"] and g[b]["strict"] for g in roots.values()),
                      "denominator":len(roots)} for name,(a,b) in kinds.items()},
       "by_root":{root:{c:{"true":v["gold"],"pred":v["prediction"],"strict":v["strict"],
                          "raw_minus_verify":v["raw_minus_verify"]} for c,v in states.items()}
                  for root,states in roots.items()}}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--base",required=True)
    p.add_argument("--adapter",required=True)
    p.add_argument("--method",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    src=jl(args.source)
    assert len(src)==15 and {x["root"] for x in src}=={"DG31","DG32","DG33"}
    hf=HFActionPolicy(model_path=args.base,adapter=args.adapter,trainable=False,
       quantization="4bit",dtype="bf16",max_prompt_tokens=4096,allow_download=False)
    hf.model.eval()
    t=time.monotonic();results=[]
    with torch.inference_mode():
        for row in src:
            assert row["NO_DG36_data_and_no_dev_labels"]
            assert assert_views_position_invariant(row["model_prompt"],row["action_menu"],
                                  row["model_prompt_swapped"],row["action_menu_swapped"])
            datum={"root":row["root"],"condition":row["condition"],"role":row["role"],
                "world":row["world"],"seen_train":row["train_source_state"],
                "gold":row["native_gold_action"],"native_state_hash":row["native_state_hash"],
                "views":{}}
            for view,prompt in (("original",row["model_prompt"]),("entity_alias",row["model_prompt_alias"])):
                acts,scores=scoring(hf,prompt,row["action_menu"])
                raw_scores=dict(zip(acts,[float(x) for x in scores.float().cpu().tolist()]))
                probs=dict(zip(acts,[float(x) for x in torch.softmax(scores.float(),dim=0).cpu().tolist()]))
                pred=exact_action(raw_scores)
                datum["views"][view]={"prediction":pred,
                   "strict":strict_correct(raw_scores,row["native_gold_action"]),
                   "candidate_scores":raw_scores,"probabilities":probs,
                   "raw_minus_verify":raw_scores[RAW]-raw_scores[VERIFY]}
            results.append(datum)
            print("V4_TRAIN_ROOT_CONTROL",args.method,row["root"],row["condition"],
                  "original",datum["views"]["original"]["prediction"],
                  "alias",datum["views"]["entity_alias"]["prediction"],flush=True)
    summ={}
    for view in ("original","entity_alias"):
        summ[view]=stats([{**{k:v for k,v in r.items() if k!="views"},**r["views"][view]} for r in results])
    args.output.mkdir(parents=True)
    (args.output/"DECISIONS.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in results),encoding="utf8")
    result={"status":"TRAIN_ROOT_EVAL_ONLY","method":args.method,
        "source_hash":sha(args.source),"n_unique_train_roots":3,
        "n_root_control_states":15,
        "seen_states":9,"unseen_controls_hold_equivalent":6,
        "views":["original","entity_alias"],"no_DG34_DG35_DG36_data_loaded":True,
        "model_candidate_forwards":len(results)*2*3,
        "runtime_seconds":round(time.monotonic()-t,2),
        "by_view":summ}
    (args.output/"SUMMARY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_TRAIN_ROOT_CONTROL_SUMMARY",json.dumps({
      v:{"strict":s["strictly_correct"],"seen":s["seen_train_states_correct"],
         "unseen":s["unseen_control_states_correct"],
         "pairs":s["pairs"]} for v,s in summ.items()},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
