"""V4 supervised history-contrast REPAIR (no cross-state DPO).

Controlled second diagnostic only if root-grouped CE does not fit TRAIN.
Uses ground-truth SAME candidate source under two actual R-ledger histories.
One root group per update. CE labels remain the native per-state action
labels. Auxiliary loss compares RAW-vs-VERIFY *logit odds* across contexts,
not a preference between actions executed in different states and not RL.

No dev or DG36 file is loaded into the optimizer.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from hashlib import sha256
import argparse,json,sys,random,time
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_v4_semantic_action import scoring,FORMAT
from kci_v4_pairwise_repair_train import (group_from_verified_states,root_schedule,RAW,VERIFY)
MARGIN=2.0
CONTRAST_WEIGHT=0.5
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def context_gap_loss(delta_risk,delta_safe):
    # This is a *supervised context sensitivity regularizer*; CE is computed
    # separately on both native states, with native labels unchanged.
    return F.softplus(MARGIN-(delta_safe-delta_risk))
def one_state_actions(hf,row):
    acts,scores=scoring(hf,row["prompt"],row["original_action_order"])
    assert acts==("ABORT","SEND_RAW_R","VERIFY_PRIVATE_R")
    ce=-torch.log_softmax(scores.float(),dim=0)[acts.index(row["chosen_action"])]
    delta=scores[acts.index(RAW)].float()-scores[acts.index(VERIFY)].float()
    return ce,delta,acts,scores
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--base",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=96)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--lr",type=float,default=2e-5)
    args=p.parse_args()
    assert args.steps==96 and args.seed==7 and args.lr==2e-5
    if args.output.exists():raise FileExistsError(args.output)
    grouped=group_from_verified_states(args.train)
    schedule=root_schedule(grouped,args.seed,args.steps)
    random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    hf=HFActionPolicy(model_path=args.base,trainable=True,
        quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
        lora_rank=8,lora_alpha=16,gradient_checkpointing=True,
        allow_download=False,temperature=1.,explore_epsilon=0.)
    opt=torch.optim.AdamW(hf.trainable_parameters(),lr=args.lr,weight_decay=0.)
    start=time.monotonic();curve=[];count=0
    print("V4_HISTORY_CONTRAST_START","roots",len(grouped),"states",9,flush=True)
    for i,root in enumerate(schedule):
        hf.model.train();opt.zero_grad(set_to_none=True)
        docs=grouped[root]
        # Retain only two histories to calculate a true context-gap gradient.
        # Gradients from risky and safe are simultaneous, non-cancelling
        # w.r.t. relative contrast. Always use signed existing native labels.
        riskce,riskdelta,ra,rs=one_state_actions(hf,docs["risky"])
        safece,safedelta,sa,ss=one_state_actions(hf,docs["safe"])
        relative=context_gap_loss(riskdelta,safedelta)
        weighted=(0.5*riskce+0.25*safece+CONTRAST_WEIGHT*relative)
        assert torch.isfinite(weighted)
        weighted.backward()
        count+=len(ra)+len(sa)
        del weighted,rs,ss
        # The authorized positive is a same-state native CE, not a fake
        # cross-state DPO pair. Clear graph after each backward.
        authce,authdelta,aa,ascores=one_state_actions(hf,docs["authorized"])
        (0.25*authce).backward()
        count+=len(aa)
        grad=torch.nn.utils.clip_grad_norm_(hf.trainable_parameters(),1.,error_if_nonfinite=True)
        assert torch.isfinite(grad)
        opt.step()
        stats={"step":i+1,"root":root,"risk_CE":float(riskce.detach()),
            "safe_CE":float(safece.detach()),"authorized_CE":float(authce.detach()),
            "contrastive_context_gap_loss":float(relative.detach()),
            "actual_raw_minus_verify_risky":float(riskdelta.detach()),
            "actual_raw_minus_verify_safe":float(safedelta.detach()),
            "actual_raw_minus_verify_authorized":float(authdelta.detach()),
            "gradient_norm_preclip":float(grad),"batch_independent_states":3}
        curve.append(stats)
        if i==0 or (i+1)%8==0:
            print("V4_HISTORY_CONTRAST",i+1,root,"contrast",round(stats["contrastive_context_gap_loss"],4),
              "risk_diff",round(stats["actual_raw_minus_verify_risky"],3),
              "safe_diff",round(stats["actual_raw_minus_verify_safe"],3),
              "ce",round(stats["risk_CE"],3),round(stats["safe_CE"],3),
              "grad",round(float(grad),3),flush=True)
    args.output.mkdir(parents=True)
    hf.save_adapter(args.output/"adapter")
    spec={
      "cell":"V4_supervised_history_context_contrast_not_RL",
      "method":"context_gap_ce",
      "base_model":args.base,"model":"Qwen3-4B",
      "optimizer":"AdamW","seed":args.seed,"lr":args.lr,"steps":args.steps,
      "quantization":"nf4_bf16","LoRA_rank":8,"LoRA_alpha":16,
      "LoRA_modules":["q_proj","v_proj"],
      "gradient_checkpointing":True,
      "semantic_action_interface":FORMAT,
      "not_RL":True,"not_cross_history_DPO":True,
      "no_dev_or_final_test_used_in_training":True,
      "n_independent_roots":3,"n_independent_training_states":9,
      "n_original_V4_normal_and_swapped_records":18,
      "no_new_swaps":True,
      "batch_state_count":3,"training_data_sha256":sha(args.train),
      "objective_weights_by_condition":{"risky":0.5,"safe":0.25,"authorized":0.25},
      "contrast_loss":"softplus(gap - ((RAW-VERIFY)_safe-(RAW-VERIFY)_risky))",
      "contrast_loss_weight":CONTRAST_WEIGHT,"contrast_margin":MARGIN,
      "no_new_teacher_labels":True,
      "per_root_updates":dict(Counter(schedule)),
      "root_schedule_sha256":sha256(json.dumps(schedule,separators=(',',':')).encode()).hexdigest(),
      "train_forward_candidate_passes":count,
      "training_wall_seconds":round(time.monotonic()-start,2),
      "curve":curve
    }
    (args.output/"TRAINING.json").write_text(json.dumps(spec,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_HISTORY_CONTRAST_COMPLETE",spec["training_wall_seconds"],"s",flush=True)
if __name__=="__main__":main()
