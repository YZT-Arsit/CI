"""Action-label counterbalanced LoRA on real Qwen3-4B KCI A counterfactuals.

Each candidate evidence bundle is evaluated under BOTH A/B letter orderings,
with the SAME authorized role-visible observation and the SAME native action
identity. This cancels spurious first-letter priors while training DPO on real
frozen 4B+R native continuation preferences (no Qstar teacher).

Match an action-CE baseline on identical training pairs, model and 2 prompts.
"""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import argparse,sys,json,random,math,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_role_lora_dpo import validate_pairs

def permute_prompt(prompt,actions):
    assert actions[0]=="A_SEND_PACKAGE_A" and actions[1]=="A_SEND_PACKAGE_B",actions
    lines=prompt.splitlines()
    x=[i for i,s in enumerate(lines) if s.startswith("A. ")]
    y=[i for i,s in enumerate(lines) if s.startswith("B. ")]
    assert len(x)==len(y)==1,(len(x),len(y))
    ia,ib=x[0],y[0]
    desc_A=lines[ia][3:]
    desc_B=lines[ib][3:]
    lines[ia]="A. "+desc_B
    lines[ib]="B. "+desc_A
    reordered=list(actions)
    reordered[0],reordered[1]=reordered[1],reordered[0]
    return "\n".join(lines),tuple(reordered)

def action_log_odds(logp,opts,good,bad):
    return logp[opts.index(good)]-logp[opts.index(bad)]

def static_variants(pair):
    orig_actions=tuple(pair["all_executable_actions"])
    prompt1=pair["prompt"]
    prompt2,actions2=permute_prompt(prompt1,orig_actions)
    good,bad=pair["chosen_action"],pair["rejected_action"]
    for opt in (orig_actions,actions2):
        assert opt.count(good)==opt.count(bad)==1
    return ((prompt1,orig_actions),(prompt2,actions2))

def one_loss(policy,rec,refs,objective,beta):
    signals=[]
    preferred_scores=[]
    for (prompt,opts),reference in zip(static_variants(rec),refs):
        lp=policy.log_distribution(prompt,opts)
        delta=action_log_odds(lp,opts,rec["chosen_action"],rec["rejected_action"])
        ref=reference
        signals.append(delta-ref)
        preferred_scores.append(lp[opts.index(rec["chosen_action"])])
    margin=torch.stack(signals).mean()
    if objective=="dpo":
        loss=-F.logsigmoid(beta*margin)
    elif objective=="action_ce":
        loss=-torch.stack(preferred_scores).mean()
    else:raise ValueError(objective)
    return loss,margin

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs",type=Path,required=True)
    ap.add_argument("--model-path",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--objective",choices=("dpo","action_ce"),default="dpo")
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--steps",type=int,default=64)
    ap.add_argument("--batch-size",type=int,default=2)
    ap.add_argument("--lr",type=float,default=0.00002)
    ap.add_argument("--beta",type=float,default=2.)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    random.seed(args.seed);torch.manual_seed(args.seed)
    if args.batch_size<1:raise ValueError("batch size")
    rows=validate_pairs(args.pairs,"RecordsAgent")
    assert len(rows)>=4
    policy=HFActionPolicy(model_path=args.model_path,trainable=True,
                          quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
                          lora_rank=8,lora_alpha=16,gradient_checkpointing=True,
                          allow_download=False,temperature=1.,explore_epsilon=0.)
    print("DPO_BASE_LOADED","n_pairs",len(rows),"objective",args.objective,
          "model",policy.policy_id,flush=True)
    # The base reference is computed once; no teacher training-set leak
    # or eval labels are needed to obtain the frozen pre-adaptation log odds.
    ref=[]
    for rec in rows:
        signs=[]
        for prompt,opts in static_variants(rec):
            logp=policy.reference_log_distribution(prompt,opts)
            signs.append(action_log_odds(logp,opts,rec["chosen_action"],
                                         rec["rejected_action"]).detach())
        ref.append(tuple(signs))
    optimizer=torch.optim.AdamW(policy.trainable_parameters(),lr=args.lr,weight_decay=0.)
    curve=[]
    for step in range(args.steps):
        policy.model.train()
        optimizer.zero_grad(set_to_none=True)
        idx=[(step*args.batch_size+j)%len(rows) for j in range(args.batch_size)]
        vals=[one_loss(policy,rows[i],ref[i],args.objective,args.beta) for i in idx]
        loss=torch.stack([a[0] for a in vals]).mean()
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(),1.,
                                                 error_if_nonfinite=True)
        optimizer.step()
        item={"step":step+1,"loss":float(loss.detach()),
              "mean_counterbalanced_ref_relative_margin":
                 float(torch.stack([a[1].detach() for a in vals]).mean()),
              "preclip_gradient_norm":float(gradient)}
        curve.append(item)
        if (step+1) in (1,5,10,20,30,40,50,64) or (step+1)%25==0:
            print("BALANCED_A_TRAIN",args.objective,args.seed,step+1,
                  "loss",round(item["loss"],5),
                  "margin",round(item["mean_counterbalanced_ref_relative_margin"],5),
                  "grad",round(item["preclip_gradient_norm"],5),flush=True)
    out=args.output
    out.mkdir(parents=True)
    policy.save_adapter(out/"adapter")
    summary={"status":"ACTUAL_QWEN3_4B_COUNTERBALANCED_A_ROLE_LORA_TRAIN",
             "objective":args.objective,
             "training_pairs":len(rows),
             "training_roots":sorted(set(rec["root"] for rec in rows)),
             "seed":args.seed,"steps":args.steps,"batch_size":args.batch_size,
             "lr":args.lr,"beta":args.beta,"reference_frozen_base":True,
             "both_A_B_action_label_orders":True,
             "train_labels_real_Qwen3_4B_R_repaired_native_CCE":True,
             "native_CI_unchanged":True,
             "dataset_sha256":sha256(args.pairs.read_bytes()).hexdigest(),
             "train_curve":curve}
    (out/"TRAINING.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("BALANCED_A_COMPLETED",args.objective,args.seed,flush=True)
if __name__=="__main__":
    main()
