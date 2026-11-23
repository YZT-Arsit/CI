"""Fixed three-arm Qwen3-4B study: CE vs same-state DPO vs joint history CE.

Not a new invented algorithm. Candidate C imports the previous V4
context_gap_loss and adds Hold/authorized constraints. Identical FIVE actor-
legal true native states from each TRAIN root for every method/optimizer step;
no extra conditions, flips, aliases, or oracle Qstar labels for C.
All methods start from one unchanged full-flow shared Qwen3-4B SFT adapter.

The native risky condition may have no task-success path under a tight deadline:
teacher VERIFY is CI-first (no RAW violation) but NEVER counted as task
success. Outcome has to be assessed from native full-agent trajectories.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import argparse,json,random,time,sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_cce_phase2_frozen4b_rollout import model_candidate_prompt
from kci_v4_history_contrast_train import context_gap_loss
from kci_v4_pairwise_repair_train import root_schedule
DATA_ROOT=ROOT/"artifacts/20261124_joint_history_supervision_v1"
CASES=("risky","safe","hold","authorized","equivalent")
RAW="SEND_RAW_TARGET";VERIFY="VERIFY_PRIVATE_TARGET";ABORT="ABORT"
ARMS=("action_ce","same_state_dpo","joint_flip_hold_ce")
SEEDS=(7,1234,2025)
STEPS=28
LR=2e-5
BETA=2.0
L_FLIP=0.5
L_AUTH=0.5
L_HOLD=0.1
L_EQUIV=0.1
def sha(p):
    digest=sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(1048576),b""):digest.update(c)
    return digest.hexdigest()
def get_data(path):
    rows=[json.loads(x) for x in Path(path).open(encoding="utf8") if x.strip()]
    assert len(rows)==35 and {r["root"] for r in rows}=={
      "P201","P202","P203","P205","P206","P207","P208"}
    grouped=defaultdict(dict)
    for x in rows:
        assert x["split"]=="train" and x["stage"]=="release"
        assert {RAW,VERIFY,ABORT}<=set(x["available_native_actions"])
        assert set(x["available_native_actions"])<={RAW,VERIFY,ABORT,"SEND_RAW_BROKER"}
        assert x["not_a_CCE_longterm_teacher"]
        assert x["native_gold"] in (RAW,VERIFY)
        assert x["same_state_rejected"] in (RAW,VERIFY) and x["native_gold"]!=x["same_state_rejected"]
        assert x["condition"] not in grouped[x["root"]]
        assert x["actor_visible_observation"]["legal_actions"]==x["available_native_actions"]
        grouped[x["root"]][x["condition"]]=x
    assert len(grouped)==7 and all(set(grouped[k])==set(CASES) for k in grouped)
    return dict(sorted(grouped.items()))
def load_frozen_shared_trainable(base,adapter):
    # Important: HFActionPolicy fresh LoRA path disallows adapter warm-start,
    # but the shared PEFT wrapper already contains exactly the ONE adapter
    # we want to update. Unfreeze only this adapter, no role router.
    hf=HFActionPolicy(model_path=base,adapter=adapter,trainable=False,
        quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
        allow_download=False,temperature=1.,explore_epsilon=0.)
    for n,p in hf.model.named_parameters():
        p.requires_grad_("lora_" in n)
    trainable=[p for p in hf.model.parameters() if p.requires_grad]
    assert trainable and sum(p.numel() for p in trainable)>10000
    assert all("lora_" in n for n,p in hf.model.named_parameters() if p.requires_grad)
    hf.model.config.use_cache=False
    hf.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    hf.model.enable_input_require_grads()
    hf.trainable=True
    return hf,trainable
def scores(hf,ob):
    actions=tuple(sorted(ob["legal_actions"]))
    assert {ABORT,RAW,VERIFY}<=set(actions)
    assert set(actions)<={ABORT,RAW,VERIFY,"SEND_RAW_BROKER"}
    values=[]
    for a in actions:
        prompt=model_candidate_prompt(ob,a)
        pair=hf.log_distribution(prompt,("ACCEPT","REJECT"))
        values.append(pair[0]-pair[1])
    return actions,torch.stack(values)
def logp(scores):
    return torch.log_softmax(scores.float(),dim=0)
def delta(actions,scores):
    return scores[actions.index(RAW)].float()-scores[actions.index(VERIFY)].float()
def matched_root_schedule(data,seed=7):
    # The exact schedule is from existing root_schedule, seed-specific,
    # same method-independent order.
    return root_schedule(data,seed,STEPS)
def symmetric_kl_logits(left,right):
    p=F.softmax(left.float(),dim=0)
    q=F.softmax(right.float(),dim=0)
    return 0.5*((p*(p.clamp_min(1e-10).log()-q.clamp_min(1e-10).log())).sum()+
                (q*(q.clamp_min(1e-10).log()-p.clamp_min(1e-10).log())).sum())

def joint_loss(chosen_ce,ds,all_scores):
    # Prior history-contrast already implemented relative RAW/VERIFY gap.
    # New C adds *only* controlled authorization and full-distribution
    # HOLD/equivalent behavior invariance; not new scenarios or labels.
    flip=context_gap_loss(ds["risky"],ds["safe"])
    auth=context_gap_loss(ds["risky"],ds["authorized"])
    hold=symmetric_kl_logits(all_scores["safe"],all_scores["hold"])
    equiv=symmetric_kl_logits(all_scores["safe"],all_scores["equivalent"])
    return chosen_ce+L_FLIP*flip+L_AUTH*auth+L_HOLD*hold+L_EQUIV*equiv, {
        "flip":flip,"authorized":auth,"hold":hold,"equivalent":equiv}
def compute_init_reference(hf,groups):
    # Same initialized shared SFT adapter, not disabled-adapter pretrained base.
    # CACHE only same-state RAW/VERIFY frozen log probability differences.
    ref={}
    hf.model.eval()
    with torch.inference_mode():
        for root,rows in groups.items():
            for cond,row in rows.items():
                acts,ss=scores(hf,row["actor_visible_observation"])
                p=logp(ss)
                ref[(root,cond)]=float(p[acts.index(row["native_gold"])]-p[acts.index(row["same_state_rejected"])])
    hf.model.train()
    assert len(ref)==35
    return ref
def train():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--base",required=True)
    p.add_argument("--init-adapter",required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    data=get_data(args.train)
    assert args.train.resolve()==(DATA_ROOT/"TRAIN_FIVE_CONTEXTS.jsonl").resolve()
    init_file=Path(args.init_adapter)/"adapter_model.safetensors"
    assert init_file.is_file()
    init_sha=sha(init_file)
    random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    hf,params=load_frozen_shared_trainable(args.base,args.init_adapter)
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.)
    schedule=matched_root_schedule(data,args.seed)
    t=time.monotonic();curve=[];refs={};ref_calls=0
    if args.arm=="same_state_dpo":
        refs=compute_init_reference(hf,data)
        ref_calls=sum(len(row["available_native_actions"]) for group in data.values() for row in group.values())
    count=0;observed=Counter()
    print("HISTORY_JOINT_STUDY_START",args.arm,args.seed,"steps",STEPS,"states",35,
          "init_sha",init_sha,flush=True)
    for i,root in enumerate(schedule):
        hf.model.train()
        opt.zero_grad(set_to_none=True)
        per={};ds={};all_scores={};ce_terms=[];dpo_terms=[]
        for cond in CASES:
            row=data[root][cond]
            acts,ss=scores(hf,row["actor_visible_observation"])
            lp=logp(ss)
            ce=-lp[acts.index(row["native_gold"])]
            ce_terms.append(ce)
            if args.arm=="same_state_dpo":
                current=lp[acts.index(row["native_gold"])]-lp[acts.index(row["same_state_rejected"])]
                dpo_terms.append(-F.logsigmoid(BETA*(current-refs[(root,cond)])))
            ds[cond]=delta(acts,ss)
            all_scores[cond]=ss
            per[cond]={"ce":float(ce.detach()),"gold":row["native_gold"],
                "raw_minus_verify":float(ds[cond].detach()),
                "predicted":acts[int(ss.detach().argmax())]}
            count+=len(acts);observed[(root,cond)]+=1
        mean_ce=torch.stack(ce_terms).mean()
        if args.arm=="action_ce":loss=mean_ce;extras={}
        elif args.arm=="same_state_dpo":loss=torch.stack(dpo_terms).mean();extras={}
        else:loss,extras=joint_loss(mean_ce,ds,all_scores)
        assert torch.isfinite(loss)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        assert torch.isfinite(grad)
        opt.step()
        sample={"step":i+1,"root":root,"loss":float(loss.detach()),
             "mean_CE_diagnostic":float(mean_ce.detach()),
             "grad_norm_preclip":float(grad),
             "samples":[per[k] for k in CASES],
             "joint_penalties":{k:float(v.detach()) for k,v in extras.items()}}
        curve.append(sample)
        if i==0 or (i+1)%7==0:
            print("HISTORY_JOINT_TRAIN",args.arm,args.seed,"update",i+1,
                  "loss",round(sample["loss"],4),"meanCE",round(sample["mean_CE_diagnostic"],4),
                  "risk_delta",round(per["risky"]["raw_minus_verify"],3),
                  "safe_delta",round(per["safe"]["raw_minus_verify"],3),
                  "grad",round(float(grad),3),flush=True)
    args.output.mkdir(parents=True)
    hf.model.save_pretrained(args.output/"adapter",safe_serialization=True)
    assert (args.output/"adapter/adapter_model.safetensors").exists()
    summary={
      "arm":args.arm,"seed":args.seed,"optimizer":"AdamW","lr":LR,
      "updates":STEPS,"states_per_update":len(CASES),"training_roots":list(data),
      "n_true_unique_training_states":35,
      "same_35_states_same_optimizer_schedule_across_all_arms":True,
      "training_contexts":list(CASES),
      "objective":"same_state_DPO_beta2" if args.arm=="same_state_dpo" else (
        "CE_plus_prior_flip_contrast_plus_hold_auth_constraints" if args.arm=="joint_flip_hold_ce" else "multiclass_action_CE"),
      "reuses_existing_context_gap_loss":True,
      "not_new_algorithm":True,
      "init_shared_full_flow_SFT_model_adapter_SHA256":init_sha,
      "one_same_Qwen3_4B_pretrained_base":args.base,
      "one_shared_LoRA_all_roles":True,"LoRA_rank":8,"LoRA_alpha":16,
      "LoRA_modules":["q_proj","v_proj"],
      "native_verifier_and_reward_unchanged":True,
      "actor_prompts_have_full_authorized_recipient_history":True,
      "no_extra_histories_for_joint_arm":True,
      "not_cce_Qpi_or_oracle_Qstar":True,
      "risky_tight_verify_may_miss_deadline":True,
      "same_state_DPO_reference":"frozen exact same initial shared-SFT adapter",
      "dpo_beta":BETA if args.arm=="same_state_dpo" else None,
      "joint_weights":({"flip":L_FLIP,"auth":L_AUTH,"hold":L_HOLD,"equiv":L_EQUIV} if args.arm=="joint_flip_hold_ce" else None),
      "total_differentiable_candidate_forwards":count,
      "reference_initial_candidate_forwards":ref_calls,
      "total_forward_calls_including_reference":count+ref_calls,
      "training_data_sha256":sha(args.train),
      "root_schedule":schedule,
      "per_root_training_updates":dict(Counter(schedule)),
      "per_root_condition_example_seen":{str(k):v for k,v in observed.items()},
      "training_wall_seconds":round(time.monotonic()-t,2),
      "model_device":"GPU_4bit_NF4_BF16",
      "dev_and_final_test_not_read":True,
      "curve":curve}
    (args.output/"TRAINING.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("HISTORY_JOINT_TRAIN_COMPLETE",args.arm,args.seed,"seconds",summary["training_wall_seconds"],
          "train_candidate_forwards",count,"ref_candidate_forwards",ref_calls,flush=True)
if __name__=="__main__":train()
