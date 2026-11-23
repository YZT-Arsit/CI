"""Measured V4-4B experiment input-token and forward/update accounting.

No model gradients, Q* reading, test data or forward passes. Uses the
already-frozen actual train update schedule from each TRAINING.json.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse,json
from transformers import AutoTokenizer
from kci_v4_train_4b import ACTION_INSTRUCTION

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf8"))
def groups(path):
    return [json.loads(s) for s in Path(path).open(encoding="utf8") if s.strip()]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--R-adapter",type=Path,required=True)
    args=p.parse_args()
    path=args.dir
    out=path/"COMPUTE_AND_INITIALIZATION.json"
    if out.exists():raise FileExistsError(out)
    ce=read(path/"train_release_actionCE_s7/TRAINING.json")
    dp=read(path/"train_release_DPO_s7/TRAINING.json")
    ev=read(path/"eval_release_4arms_s7/SUMMARY.json")
    g=groups(path/"TRAIN_RELEASE_GROUPS.jsonl")
    assert len(g)==9
    assert ce["steps"]==dp["steps"]==96 and ce["seed"]==dp["seed"]==7
    seq1=[r["comparison_id"] for r in ce["training_curve"]]
    seq2=[r["comparison_id"] for r in dp["training_curve"]]
    assert seq1==seq2,"CE/DPO minibatch group schedule mismatch"
    tok=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True,
                                      trust_remote_code=False)
    lengths={}
    for group in g:
        sizes=[]
        for r in group["paired_option_records"]:
            prompt=r["prompt"]+ACTION_INSTRUCTION
            if tok.chat_template:
                seq=tok.apply_chat_template([{"role":"user","content":prompt}],
                    tokenize=True,add_generation_prompt=True)
            else:seq=tok.encode(prompt,add_special_tokens=True)
            assert 1<=len(seq)<=4096
            sizes.append(len(seq))
        lengths[group["group_id"]]=sizes
    training_total=sum(sum(lengths[x]) for x in seq1)
    unique_prompt_tokens=sum(sum(x) for x in lengths.values())
    report={
       "status":"MEASURED_MODEL_INPUT_AND_ACTUAL_UPDATE_ACCOUNTING",
       "data_group_sha256":sha(path/"TRAIN_RELEASE_GROUPS.jsonl"),
       "model_base_config_sha256":sha(args.model/"config.json"),
       "model_tokenizer_config_sha256":sha(args.model/"tokenizer_config.json"),
       "historic_R_adapter_weights_sha256":sha(args.R_adapter/"adapter_model.safetensors"),
       "CE_trained_adapter_weights_sha256":sha(path/"train_release_actionCE_s7/adapter/adapter_model.safetensors"),
       "DPO_trained_adapter_weights_sha256":sha(path/"train_release_DPO_s7/adapter/adapter_model.safetensors"),
       "CE_and_DPO_same_group_schedule_sha256":
          sha256(json.dumps(seq1,separators=(',',':')).encode()).hexdigest(),
       "independent_training_comparisons":len(g),
       "two_original_order_rows_each":True,
       "actual_optimizer_updates_per_arm":ce["steps"],
       "differentiable_forward_backward_prompts_per_arm":2*ce["steps"],
       "measured_pre_chat_LM_tokenized_prompt_tokens_per_arm":training_total,
       "DPO_frozen_base_reference_forward_prompts":2*len(g),
       "DPO_frozen_reference_tokenized_prompt_tokens":unique_prompt_tokens,
       "original_train_option_prompt_token_lengths_by_group":lengths,
       "CE_real_training_wall_seconds":ce["wall_seconds"],
       "DPO_real_training_wall_seconds":dp["wall_seconds"],
       "eval_forward_passes_by_method":ev["forward_calls_by_method"],
       "eval_tokenized_prompt_tokens_by_method":ev["prompt_tokens_sum_by_method"],
       "eval_wall_seconds":ev["total_eval_wall_seconds"],
       "GPU_type":"RTX 4090 D (24GiB)",
       "peak_GPU_VRAM_MiB":"not instrumented; monitoring observed ~5874 MiB during training",
       "no_GFLOPS_claim_without_profiling":True,
       "no_final_test_DG36_used":True
    }
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_COMPUTE",json.dumps({k:v for k,v in report.items() if k!="original_train_option_prompt_token_lengths_by_group"},ensure_ascii=False)[:5500])
if __name__=="__main__":main()
