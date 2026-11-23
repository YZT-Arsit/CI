"""Real V4 native full-agent diagnostic for FROZEN paired-training repair.

NO training and NO dev-based tuning. Uses one shared Qwen3-4B LoRA:
  controlled_release: original V4 signed A/B prefix, all subsequent RELEASE
                      decisions by frozen 4B, identical other native tools;
  autonomous_shared: same adapter for all roles from native initial state.
Runs only two previously declared development roots DG34/DG35.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,sys,time,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_v4_semantic_action import scoring,semantic_v4_candidate
from kci_v4_semantic_full_agent import run_env,spec_table,read
class SingleSharedRepair:
    def __init__(self,model_path,adapter):
        self.hf=HFActionPolicy(model_path=model_path,adapter=adapter,trainable=False,
                  quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
                  allow_download=False)
        self.hf.model.eval()
        self.calls=0
        self.tokenized_inputs=0
    @torch.inference_mode()
    def distribution(self,prompt,actions,method):
        acts,logits=scoring(self.hf,prompt,actions)
        scores={act:float(s) for act,s in zip(acts,logits.float().cpu().tolist())}
        probs=torch.softmax(logits.float(),dim=0).cpu().tolist()
        self.calls+=len(acts)
        self.tokenized_inputs+=sum(self.hf._input_ids(semantic_v4_candidate(prompt,actions,a)).numel() for a in acts)
        return dict(zip(acts,[float(x) for x in probs])),scores
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--adapter",required=True)
    p.add_argument("--train-gate",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    gate=json.loads(args.train_gate.read_text(encoding="utf8"))
    assert gate["fit_gate_passed"] and gate["true_flip_both_sides_correct_roots"]==3
    assert gate["DG34_DG35_DG36_NOT_READ"]
    specs=spec_table(args.data_root)
    worlds=read(args.data_root/"DEV_WORLD_CONFIGS.jsonl")
    assert len(worlds)==10 and {w["root_id"] for w in worlds}=={"DG34","DG35"}
    policy=SingleSharedRepair(args.base,args.adapter)
    all_rows=[];t=time.monotonic()
    args.output.mkdir(parents=True)
    with (args.output/"WORLD_EPISODES.jsonl").open("x",encoding="utf8") as f:
        for protocol in ("controlled_release","autonomous_shared"):
            for view in (("native","entity_alias") if protocol=="controlled_release" else ("native",)):
                for world in worlds:
                    result=run_env(specs,world,"grouped_ce_repair",view,protocol,policy)
                    assert result["root"] in ("DG34","DG35") and result["split"]=="dev"
                    assert result["all_actions_attempted_against_native_env"]
                    all_rows.append(result)
                    f.write(json.dumps(result,sort_keys=True,ensure_ascii=False)+"\n")
                    f.flush()
                    print("V4_REPAIR_NATIVE",protocol,view,world["world_id"],
                          "success",result["native_actual_success"],
                          "violation",result["native_actual_violation"],
                          "abort",result["native_abort"],
                          "unknown",result["native_incomplete_due_rejected_or_UNKNOWN_action"],
                          "n_model_decisions",result["model_decision_count"],flush=True)
    summary={}
    for protocol in ("controlled_release","autonomous_shared"):
        summary[protocol]={}
        for view in (("native","entity_alias") if protocol=="controlled_release" else ("native",)):
            x=[r for r in all_rows if r["protocol"]==protocol and r["role_view"]==view]
            assert len(x)==10
            by_root=defaultdict(dict)
            for r in x:by_root[r["root"]][r["condition"]]=r
            summary[protocol][view]={
                "total_worlds":len(x),"independent_business_roots":len(by_root),
                "safe_completions":sum(r["native_actual_success"] for r in x),
                "CI_violations":sum(r["native_actual_violation"] for r in x),
                "aborted":sum(r["native_abort"] for r in x),
                "unknown_incomplete":sum(r["native_incomplete_due_rejected_or_UNKNOWN_action"] for r in x),
                "native_actual_steps_total":sum(r["native_execution_steps"] for r in x),
                "mean_native_steps_total":sum(r["native_execution_steps"] for r in x)/len(x),
                "model_choices":sum(r["model_decision_count"] for r in x),
                "decision_roles":dict(sum((Counter(r["model_roles"]) for r in x),Counter())),
                "risky_state_violations":sum(v["risky"]["native_actual_violation"] for v in by_root.values()),
                "safe_state_completions":sum(v["safe"]["native_actual_success"] for v in by_root.values()),
                "authorization_completions":sum(v["authorized"]["native_actual_success"] for v in by_root.values()),
                "not_a_Qstar_or_Qpi_value_calculation":True,
                "autonomous_changes_upstream_histories_so_not_strict_history_pair":protocol=="autonomous_shared"}
    result={"model":"one_shared_Qwen3_4B_LoRA_grouped_CE",
        "not_trained_or_selected_on_development":True,
        "train_triplet_gate_passed_before_development":True,
        "native_verifier_rules_untouched":True,
        "DG36_unread":True,
        "n_full_native_team_episodes":len(all_rows),
        "n_model_candidate_forwards":policy.calls,
        "total_model_tokenized_input_tokens":policy.tokenized_inputs,
        "runtime_seconds":round(time.monotonic()-t,2),
        "by_protocol":summary}
    (args.output/"SUMMARY.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_REPAIR_NATIVE_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
