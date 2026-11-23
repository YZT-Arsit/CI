"""Post-hoc matched-seed fresh Qwen3-4B LoRA initialization control.

Reconstruct the exact *initialization protocol* used by first-stage Action
SFT: fresh base NF4/BF16, LoRA r=8/alpha=16 q_proj/v_proj, global seed7.
Save BEFORE any optimizer steps. Does not change trained SFT checkpoint.
This untrained initialization is a capability control, not a CCE teacher.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import argparse,json,random,time,sys,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--sft-training",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    spec=json.loads(args.sft_training.read_text(encoding="utf8"))
    assert (spec["seed"],spec["sampled_sft_steps"],spec["lr"])==(7,160,2e-5)
    assert spec["lora_rank"]==8 and spec["lora_alpha"]==16
    assert spec["base"]==args.base
    random.seed(7);torch.manual_seed(7)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(7)
    start=time.monotonic()
    policy=HFActionPolicy(model_path=args.base,trainable=True,
       quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
       lora_rank=8,lora_alpha=16,gradient_checkpointing=True,
       allow_download=False,temperature=1.,explore_epsilon=0.)
    assert "fresh-lora" in policy.policy_id
    args.output.mkdir(parents=True)
    policy.save_adapter(args.output/"adapter")
    result={
      "cell":"Untrained fresh shared LoRA before complete-flow SFT",
      "posthoc_initialization_control_not_blinded_method_selection":True,
      "exact_same_documented_init_parameters_as_SFT":True,
      "seeds":{"random":7,"torch_cpu":7,"torch_cuda_all":7},
      "base_model":args.base,
      "same_NF4_BF16":True,"LoRA_r":8,"LoRA_alpha":16,
      "LoRA_targets":["q_proj","v_proj"],
      "same_gradient_checkpointing_configuration":True,
      "same_single_shared_model_for_all_roles":True,
      "no_trainable_weights_ever_optimized":True,
      "saved_initial_adapter_sha256":sha(args.output/"adapter/adapter_model.safetensors"),
      "source_original_SFT_training_sha256":sha(args.sft_training),
      "model_initialization_seconds":round(time.monotonic()-start,2),
      "no_DEV_or_final_test_loaded":True}
    (args.output/"INITIALIZATION.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_FRESH_SFT_INIT_FROZEN",json.dumps(result,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
