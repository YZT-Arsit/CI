"""Matched three-arm upstream A optimization FROM EXACT SAME shared-SFT LoRA.

This is a controlled SIGNAL-IDENTITY study: on accepted TRAIN states the
native successful Action CE, terminal propagation, and CCE teacher labels
are identical 7/7. Refit separately to confirm no independent CCE effect
can arise under the same optimizer and initial weights.

Does NOT train from Qstar as current policy Qpi. No dev/test data read.
ABORT remains available. All roles share ONE PEFT LoRA (no role routing).
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
from hashlib import sha256
import argparse,json,random,sys,time,torch
from torch.nn.functional import log_softmax
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_cce_full_flow_sft_train import candidate_scores
from kci_cce_phase2_frozen4b_rollout import model_candidate_prompt
FROZEN_SEED=7
STEPS=48
LR=2e-5
ARM_TO_LABEL={
  "action_ce":"action_ce_witness",
  "terminal":"terminal_K8",
  "CCE":"CCE_confirmed"}
def sha(p):
    h=sha256()
    with Path(p).open("rb") as f:
        for ch in iter(lambda:f.read(1048576),b""):h.update(ch)
    return h.hexdigest()
def read(p):return [json.loads(x) for x in Path(p).open(encoding="utf8") if x.strip()]
def schedule(rows,seed=7,steps=48):
    random_rng=random.Random(seed)
    order=[]
    while len(order)<steps:
        ix=list(range(len(rows)))
        random_rng.shuffle(ix)
        order.extend(ix)
    return order[:steps]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",required=True)
    p.add_argument("--shared-sft-adapter",type=Path,required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--pre-reg",type=Path,required=True)
    p.add_argument("--arm",choices=tuple(ARM_TO_LABEL),required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    meta=json.loads(args.pre_reg.read_text(encoding="utf8"))
    assert meta["status"]=="FROZEN_MATCHED_THREE_ARM_A_SUPERVISION"
    assert meta["source_labels_SHA256"]==sha(args.train)
    assert meta["previous_SFT_adapter_hash"]==sha(args.shared_sft_adapter/"adapter_model.safetensors")
    assert meta["frozen_training_steps_per_arm"]==STEPS
    rows=sorted(read(args.train),key=lambda x:x["root"])
    assert len(rows)==7 and len({x["root"] for x in rows})==7
    assert all(x["split"]=="train" and x["stage"]=="A_SEND" and x["actor"]=="RecordsAgent" for x in rows)
    assert all(x["both_current_A_deliveries_are_native_VALID"] and "ABORT" in x["allowed_native_A_actions"] for x in rows)
    assert all(x["teacher_actions"]["action_ce_witness"]==x["teacher_actions"]["terminal_K8"]==
               x["teacher_actions"]["CCE_confirmed"] for x in rows)
    chosen_key=ARM_TO_LABEL[args.arm]
    idx=schedule(rows,FROZEN_SEED,STEPS)
    random.seed(FROZEN_SEED);torch.manual_seed(FROZEN_SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(FROZEN_SEED)
    # HFActionPolicy loads inference adapter with identical NF4/BF16 base and
    # SAME SFT weights in each of the three independent fresh processes.
    hf=HFActionPolicy(model_path=args.base,adapter=str(args.shared_sft_adapter),
        trainable=False,quantization="4bit",dtype="bf16",
        max_prompt_tokens=4096,allow_download=False,
        temperature=1.,explore_epsilon=0.)
    # Functional PEFT adapter-continuation training, intentionally restricted
    # to original SINGLE shared adapter weights, not role-specific routing.
    hf.model.enable_input_require_grads()
    hf.model.gradient_checkpointing_enable()
    hf.model.config.use_cache=False
    params=[]
    for name,weight in hf.model.named_parameters():
        is_lora="lora_A.default" in name or "lora_B.default" in name
        weight.requires_grad_(is_lora)
        if is_lora:params.append(weight)
    assert params and len(params)>=50
    assert not any(p.requires_grad for name,p in hf.model.named_parameters()
                   if "lora_A.default" not in name and "lora_B.default" not in name)
    hf.trainable=True
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.)
    t=time.monotonic();curve=[];n_forward=0
    print("CCE_MATCHED_A_TRAIN_START",args.arm,"sft_hash",meta["previous_SFT_adapter_hash"],
          "train_roots",len(rows),"updates",STEPS,flush=True)
    for step,i in enumerate(idx):
        hf.model.train();opt.zero_grad(set_to_none=True)
        row=rows[i]
        actions,scores=candidate_scores(hf,row["role_observation"])
        assert len(actions)==3 and "ABORT" in actions
        gold=actions.index(row["teacher_actions"][chosen_key])
        loss=-log_softmax(scores.float(),dim=0)[gold]
        assert torch.isfinite(loss)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        assert torch.isfinite(grad)
        opt.step()
        n_forward+=len(actions)
        pred=actions[int(scores.detach().argmax())]
        curve.append({"step":step+1,"root":row["root"],
          "teacher_source":chosen_key,"target_action":actions[gold],
          "preupdate_prediction":pred,"loss":float(loss.detach()),
          "grad_before_clip":float(grad)})
        if step==0 or (step+1)%8==0:
            print("CCE_MATCHED_A_TRAIN",args.arm,step+1,row["root"],
               "loss",round(float(loss.detach()),5),
               "guess",pred,"gold",actions[gold],
               "grad",round(float(grad),3),flush=True)
    args.output.mkdir(parents=True)
    hf.save_adapter(args.output/"adapter")
    data={
      "experiment":"Matched same-label three-A-optimizer negative control",
      "method":args.arm,"teacher_key":chosen_key,
      "same_actor_visible_input":True,"n_training_states":len(rows),
      "TRAIN_roots":[r["root"] for r in rows],
      "all_three_teacher_labels_identical_across_7_states":True,
      "initial_model_SFT_adapter_sha256":meta["previous_SFT_adapter_hash"],
      "source_teacher_file_sha256":sha(args.train),
      "previous_trained_base":args.base,"same_single_shared_LoRA":True,
      "ABORT_kept_in_model_action_menu":True,
      "not_DPR_or_online_RL":True,
      "optimized_only_default_shared_LoRA":True,
      "method_optimizer":"CE on action semantic candidate softmax (identical for all three)",
      "LoRA_r":8,"LoRA_alpha":16,
      "optimizer":"AdamW","weight_decay":0.,"seed":FROZEN_SEED,
      "steps":STEPS,"batch":1,"lr":LR,
      "schedule_sha256":sha256(json.dumps(idx,separators=(",",":")).encode()).hexdigest(),
      "sample_counts_by_root":dict(Counter(rows[i]["root"] for i in idx)),
      "n_candidate_forward_calls":n_forward,
      "wall_training_seconds":round(time.monotonic()-t,2),
      "real_CCE_policy_Qpi_based_preference_only":args.arm=="CCE",
      "native_CE_witness_teacher_not_Qstar_or_model_Qpi":args.arm=="action_ce",
      "no_DEV_or_FINAL_data_read_or_trained":True,
      "curve":curve,
    }
    (args.output/"TRAINING.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_MATCHED_A_TRAIN_DONE",args.arm,
          "adapter_sha256",sha(args.output/"adapter/adapter_model.safetensors"),
          "seconds",data["wall_training_seconds"],flush=True)
if __name__=="__main__":main()
