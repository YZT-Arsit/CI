"""4B action learning with identical inputs/budget; only teacher labels change.
Paired label counterbalancing neutralizes raw-first letter bias.
"""
from __future__ import annotations
import argparse,hashlib,json,random,sys,time
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy

RAW,VERIFY="SEND_RAW_R","VERIFY_PRIVATE_R"
def variants(r):
    opts=tuple(r["all_executable_actions"])
    assert opts[:2]==(RAW,VERIFY),opts
    prompt=r["prompt"]
    lines=prompt.splitlines()
    ix=[i for i,s in enumerate(lines) if s.startswith("A. ")]
    iy=[i for i,s in enumerate(lines) if s.startswith("B. ")]
    assert len(ix)==len(iy)==1
    i,j=ix[0],iy[0]
    swapped=lines[:]
    swapped[i]="A. "+lines[j][3:]
    swapped[j]="B. "+lines[i][3:]
    so=list(opts);so[0],so[1]=so[1],so[0]
    return [(prompt,opts),("\n".join(swapped),tuple(so))]
def validate(path):
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    assert len(rows)==32
    assert len({x["root"] for x in rows})==4
    assert set(x["split"] for x in rows)=={"train"}
    for r in rows:
        assert r["stage"]=="release"
        assert r["chosen_action"] in (RAW,VERIFY)
        assert r["rejected_action"] in (RAW,VERIFY)
        assert r["chosen_action"]!=r["rejected_action"]
        assert r["teacher_is_native_verifier_not_Qpi"]
        assert r["not_Qstar_oracle_teacher"]
        assert len(r["native_evidence"])==2
        for prompt,opts in variants(r):
            assert opts[opts.index(r["chosen_action"])]==r["chosen_action"]
            assert "native_state" not in prompt and "Qstar" not in prompt
    return rows
def one_loss(policy,row,refs,objective,beta):
    losses=[]
    margins=[]
    for (prompt,opts),ref in zip(variants(row),refs):
        lp=policy.log_distribution(prompt,opts)
        c=lp[opts.index(row["chosen_action"])]
        d=lp[opts.index(row["rejected_action"])]
        delta=c-d
        margins.append(delta-ref)
        if objective=="dpo":
            losses.append(-F.logsigmoid(beta*(delta-ref)))
        else:
            losses.append(-c)
    return torch.stack(losses).mean(),torch.stack(margins).mean()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pairs",type=Path,required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--objective",choices=["dpo","action_ce"],default="dpo")
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--steps",type=int,default=64)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--beta",type=float,default=2.0)
    args=p.parse_args()
    assert not args.output.exists()
    assert args.steps>0 and 1<=args.batch<=8
    rows=validate(args.pairs)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    policy=HFActionPolicy(model_path=args.model,trainable=True,quantization="4bit",
            dtype="bf16",max_prompt_tokens=4096,lora_rank=8,lora_alpha=16,
            gradient_checkpointing=True,allow_download=False,
            temperature=1.0,explore_epsilon=0.0)
    print("START_PAIR_TRAIN",args.objective,args.pairs.name,
          "n",len(rows),"seed",args.seed,flush=True)
    start=time.monotonic()
    refs=[]
    if args.objective=="dpo":
        for r in rows:
            with torch.inference_mode():
                refs.append(tuple(
                    (lambda lp,opts: lp[opts.index(r["chosen_action"])]-lp[opts.index(r["rejected_action"])])
                    (policy.reference_log_distribution(prompt,opts),opts).detach()
                    for prompt,opts in variants(r)))
    else:
        refs=[(torch.tensor(0.,device="cuda"),torch.tensor(0.,device="cuda"))
              for _ in rows]
    optimizer=torch.optim.AdamW(policy.trainable_parameters(),lr=args.lr,weight_decay=0.)
    curve=[]
    for step in range(args.steps):
        policy.model.train()
        optimizer.zero_grad(set_to_none=True)
        ix=[(step*args.batch+j)%len(rows) for j in range(args.batch)]
        values=[one_loss(policy,rows[i],refs[i],args.objective,args.beta) for i in ix]
        loss=torch.stack([a for a,b in values]).mean()
        assert torch.isfinite(loss)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(),1.0,error_if_nonfinite=True)
        assert torch.isfinite(grad)
        optimizer.step()
        item={"step":step+1,"loss":float(loss.detach()),
              "mean_ref_delta":float(torch.stack([b.detach() for a,b in values]).mean()),
              "preclip_grad_norm":float(grad)}
        curve.append(item)
        if step==0 or (step+1)%8==0:
            print("PAIR_TRAIN_STEP",args.objective,args.pairs.name,step+1,
                  round(item["loss"],6),round(item["mean_ref_delta"],5),
                  round(item["preclip_grad_norm"],5),flush=True)
    args.output.mkdir(parents=True)
    policy.save_adapter(args.output/"adapter")
    manifest={
        "objective":args.objective,"label_access":rows[0]["teacher_history_access"],
        "n_training_rows":len(rows),
        "n_independent_train_roots":len({r["root"] for r in rows}),
        "training_labels":dict(Counter(r["chosen_action"] for r in rows)),
        "pairs_sha256":hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "seed":args.seed,"optimizer":"AdamW","lr":args.lr,"beta":args.beta,
        "steps":args.steps,"batch":args.batch,
        "counterbalanced_raw_verify_menu":True,
        "frozen_base_reference":args.objective=="dpo",
        "base_model":args.model,"elapsed_seconds":round(time.monotonic()-start,2),
        "no_native_Qstar_teacher":True,
        "same_role_observation_for_evaluator_conditions":True,
        "train_curve":curve}
    (args.output/"TRAINING.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print("FINISHED_PAIR_TRAIN",args.objective,args.pairs.name,round(time.monotonic()-start,2),flush=True)
if __name__=="__main__":main()
