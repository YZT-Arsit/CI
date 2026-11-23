"""Diagnose native Qwen3-4B A action-token position bias via swap counterbalance.

For each SAME role observation, re-label A/B choice positions without touching
native actions or source facts; evaluate a frozen LLM twice, once per order.
This is a POLICY/DECODER diagnostic, not label generation or simulator editing.
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from collections import Counter
import sys,json,math,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_gpu_4090_eval_A_R_roles import RoutedActual4B
from kci_exact_policy_continuation_cpu import environment_public
from compare_kci_chain_methods_cpu_action_interaction import GROUP

def logits_for_A(obs,model,mode):
    _,orig=model.policy(obs,mode)
    swapped=deepcopy(obs)
    options=swapped["legal_actions"]
    i=options.index("A_SEND_PACKAGE_A")
    j=options.index("A_SEND_PACKAGE_B")
    options[i],options[j]=options[j],options[i]
    _,swap=model.policy(swapped,mode)
    log1=math.log(max(orig["A_SEND_PACKAGE_A"],1e-30))-math.log(max(orig["A_SEND_PACKAGE_B"],1e-30))
    log2=math.log(max(swap["A_SEND_PACKAGE_A"],1e-30))-math.log(max(swap["A_SEND_PACKAGE_B"],1e-30))
    content_score=(log1+log2)/2
    return {"original_gap_A_minus_B":log1,"swapped_gap_A_minus_B":log2,
            "permutation_average_gap":content_score,
            "original_action":"A" if log1>0 else "B",
            "swapped_action":"A" if log2>0 else "B",
            "calibrated_action":"A" if content_score>1e-4 else "B" if content_score< -1e-4 else "tie",
            "original_prob_A":orig["A_SEND_PACKAGE_A"],
            "swapped_prob_A":swap["A_SEND_PACKAGE_A"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True)
    ap.add_argument("--A-adapter",required=True)
    ap.add_argument("--R-adapter",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    path.mkdir(parents=True)
    model=RoutedActual4B(args.base,args.A_adapter,args.R_adapter)
    worlds=[]
    for split,rows in GROUP.items():
        for row in rows:
            e=environment_public(row)
            st,a=e.transition(e.initial(),"A_READ_VERIFIED_PACKAGES")
            assert a["verdict"]=="VALID"
            obs=e.observation(st)
            q=row["upstream_A_values_given_exact_B_continuation"]
            real_label="A" if q["PACKAGE_A"]>q["PACKAGE_B"]+1e-8 else "B" if q["PACKAGE_B"]>q["PACKAGE_A"]+1e-8 else "tie"
            cases={mode:logits_for_A(obs,model,mode) for mode in ("R_only","A_plus_R")}
            worlds.append({"split":split,"world":row["world"],"root":row["root"],
                           "source_mode":row["source_mode"],"native_Qstar_label_eval_only":real_label,
                           "scores":cases})
    summary={}
    for split in GROUP:
        xs=[r for r in worlds if r["split"]==split]
        summary[split]={}
        for method in ("R_only","A_plus_R"):
            strict=[r for r in xs if r["native_Qstar_label_eval_only"]!="tie"]
            scores=[r["scores"][method] for r in strict]
            summary[split][method]={
                "strict_states":len(strict),
                "original_Qstar_accuracy":sum(s["original_action"]==r["native_Qstar_label_eval_only"] for s,r in zip(scores,strict)),
                "counterbalanced_Qstar_accuracy":sum(s["calibrated_action"]==r["native_Qstar_label_eval_only"] for s,r in zip(scores,strict)),
                "counterbalanced_ties":sum(s["calibrated_action"]=="tie" for s in scores),
                "position_flips_under_swap":sum(s["original_action"]!=s["swapped_action"] for s in scores),
                "mean_original_log_odds":sum(s["original_gap_A_minus_B"] for s in scores)/len(scores),
                "mean_swapped_log_odds":sum(s["swapped_gap_A_minus_B"] for s in scores)/len(scores),
                "mean_counterbalanced_log_odds":sum(s["permutation_average_gap"] for s in scores)/len(scores),
            }
    result={"status":"ACTUAL_QWEN3_4B_ROLE_A_ACTION_LABEL_SWAP_DIAGNOSTIC",
            "native_worlds_unchanged":True,"source_and_A_adapters_unchanged":True,
            "Qstar_only_separate_evaluation":True,
            "full":worlds,"summary":summary,
            "model_forward_calls":model.model_calls,"token_usage":model.prompt_tokens}
    (path/"RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("LABEL_SWAP_RESULTS",json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
