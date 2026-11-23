"""One-shot V4 held-out dev evaluation for PRESELECTED training-fit model.

Reads DG34/35 only AFTER the strict train-only gate is passed. Models never
see native oracle labels. Original/swap menus are byte-identical inputs per
action, not extra samples. Entity aliases do not change native trajectories.
"""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_v4_semantic_eval import source_states,decision_model_metrics
from kci_v4_semantic_action import scoring,assert_views_position_invariant
RAW,VERIFY="SEND_RAW_R","VERIFY_PRIVATE_R"
def read(path):return [json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s.strip()]
def sha(path):return sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--adapter",required=True)
    p.add_argument("--method",required=True)
    p.add_argument("--train-gate",type=Path,required=True)
    p.add_argument("--source-data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    gate=json.loads(args.train_gate.read_text(encoding="utf8"))
    assert gate["model_method"]==args.method
    assert gate["fit_gate_passed"],"Never touch dev unless all three train-root triplets fit"
    assert gate["DG34_DG35_DG36_NOT_READ"]
    train=read(args.source_data/"TRAIN_HISTORY_AWARE.jsonl")
    dev=read(args.source_data/"DEV_DUAL_VIEW_DG34_DG35.jsonl")
    assert len(train)==9 and len(dev)==10 and set(x["root"] for x in dev)=={"DG34","DG35"}
    states=[x for x in source_states(train,dev) if x["split"]=="dev"]
    hf=HFActionPolicy(model_path=args.base,adapter=args.adapter,trainable=False,
           quantization="4bit",dtype="bf16",max_prompt_tokens=4096,allow_download=False)
    hf.model.eval()
    data=[];start=time.monotonic()
    with torch.inference_mode():
        for row in states:
            assert_views_position_invariant(row["prompt_native"],row["options"],
                                            row["swapped_prompt"],row["swapped_options"])
            rec={"root":row["root"],"world":row["world"],"condition":row["condition"],
                 "role":row["role"],"native_truth":row["gold"],
                 "native_state_hash":row["native_hash"],
                 "native_outcomes":row["native_outcomes"],
                 "views":{}}
            for name,prompt in (("native",row["prompt_native"]),("entity_alias",row["prompt_alias"])):
                actions,scores=scoring(hf,prompt,row["options"])
                probs=torch.softmax(scores.float(),dim=0).cpu().tolist()
                p=dict(zip(actions,[float(x) for x in probs]))
                s=dict(zip(actions,[float(x) for x in scores.float().cpu().tolist()]))
                action=max(p,key=p.get)
                rec["views"][name]={
                    "P":p,"candidate_scores":s,"selected":action,
                    "correct":action==row["gold"],
                    "original_and_swapped_order_same_prompt":True,
                    "RAW_minus_VERIFY":s[RAW]-s[VERIFY]}
            rec["alias_action_changed"]=rec["views"]["native"]["selected"]!=rec["views"]["entity_alias"]["selected"]
            data.append(rec)
            print("V4_REPAIR_DEV",args.method,row["root"],row["condition"],
                  "native",rec["views"]["native"]["selected"],
                  "alias",rec["views"]["entity_alias"]["selected"],flush=True)
    summaries={}
    for view in ("native","entity_alias"):
        rows=[]
        for rec in data:
            x=rec["views"][view]
            rows.append({"root":rec["root"],"condition":rec["condition"],
              "gold":rec["native_truth"],"selected":x["selected"],"P":x["P"],
              "native_outcomes":{
                a:{k:v for k,v in v.items() if k!="full_fixed_suffix_trace"}
                for a,v in rec["native_outcomes"].items()}})
        summaries[view]=decision_model_metrics(rows)
    args.output.mkdir(parents=True)
    (args.output/"DECISIONS_DG34_DG35.jsonl").write_text("".join(
        json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in data),encoding="utf8")
    report={
       "model":args.method,
       "dev_precondition_train_root3_triples_all_fit":True,
       "dev_sealed_until_fixed_method_selected_by_training_fit":True,
       "no_DG36_evaluated":True,
       "n_independent_dev_roots":2,
       "n_dev_states":10,
       "n_original_and_alias_views":2,
       "original_and_swapped_exactly_same_candidate_inputs":True,
       "original_model_Qpi_not_from_native_Qstar":True,
       "train_gate_sha256":sha(args.train_gate),
       "source_dev_data_sha256":sha(args.source_data/"DEV_DUAL_VIEW_DG34_DG35.jsonl"),
       "n_alias_action_flips":sum(r["alias_action_changed"] for r in data),
       "forward_calls":10*2*3,
       "wall_seconds":round(time.monotonic()-start,2),
       "by_view":summaries}
    (args.output/"SUMMARY.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_REPAIR_DEV_GATE",json.dumps({
       view:{"correct":v["state_actions_correct"],
          "Flip":v["pair_metrics"]["Flip"]["both_correct"],
          "Hold":v["pair_metrics"]["Hold"]["both_correct"],
          "Authorized":v["pair_metrics"]["Authorized"]["both_correct"],
          "legalRAW":v["raw_on_legitimate"],"unsafeRAW":v["raw_on_risky"],
          "success":v["native_fixed_suffix_safe_success"],
          "extra_steps":v["native_fixed_suffix_extra_steps_sum"]}
       for view,v in summaries.items()},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
