"""Fixed Qwen3-4B NF4 LoRA controlled V4 semantic-action experiments.

Three frozen arms: history-aware CE, history-aware DPO, history-blind
teacher CE. SAME actor prompts with true history for ALL arms. Objective
source is the only difference between aware/blind CE; CE/DPO change loss.
Original option swaps do NOT double the independent sample count.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse,json,random,sys,time
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_v4_semantic_action import scoring,log_action_distribution,FORMAT
def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s.strip()]
def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def load_training(path,arm):
    rows=read(path)
    assert len(rows)==9 and len(set(r["comparison_id"] for r in rows))==9
    assert Counter(r["root"] for r in rows)=={"DG31":3,"DG32":3,"DG33":3}
    assert all(r["source_split"]=="train" and r["stage"]=="release" for r in rows)
    assert all(r["canonical_actions"]==sorted(r["original_action_order"]) for r in rows)
    for r in rows:
        assert r["teacher_type"]==("V4_native_teacher_R_receipts_cleared_only" if arm=="blind_ce" else "V4_real_history_native_teacher")
        assert len(r["semantic_candidate_sha256"])==3
        assert r["chosen_action"]!=r["rejected_action"]
        assert r["chosen_action"] in r["canonical_actions"] and r["rejected_action"] in r["canonical_actions"]
    return rows

def fixed_schedule(rows,seed,steps):
    rng=random.Random(seed);pool=[];idx=[]
    for step in range(steps):
        if not pool:
            pool=list(range(len(rows)));rng.shuffle(pool)
        idx.append(pool.pop())
    return idx

def train():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--base",type=str,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--arm",choices=("aware_ce","aware_dpo","blind_ce"),required=True)
    p.add_argument("--steps",type=int,default=96)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--beta",type=float,default=2.0)
    p.add_argument("--seed",type=int,default=7)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=load_training(args.train,args.arm)
    idx=fixed_schedule(rows,args.seed,args.steps)
    random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    model=HFActionPolicy(model_path=args.base,trainable=True,
          quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
          lora_rank=8,lora_alpha=16,gradient_checkpointing=True,
          allow_download=False,temperature=1.0,explore_epsilon=0.0)
    assert "fresh-lora" in model.policy_id
    print("V4_SEMANTIC_TRAIN_START",args.arm,model.policy_id,
          "training_independent_states",len(rows),"optimizer_steps",args.steps,flush=True)
    t=time.monotonic()
    refs={}
    if args.arm=="aware_dpo":
        model.model.eval()
        with torch.no_grad():
            for r in rows:
                actions,base_lp=log_action_distribution(model,r["prompt"],r["original_action_order"],reference=True)
                refs[r["comparison_id"]]=base_lp[actions.index(r["chosen_action"])]-base_lp[actions.index(r["rejected_action"])]
        print("V4_SEMANTIC_FROZEN_BASE_REF_READY",len(refs),flush=True)
    opt=torch.optim.AdamW(model.trainable_parameters(),lr=args.lr,weight_decay=0.0)
    curve=[];forward_counts=Counter()
    for step,i in enumerate(idx):
        row=rows[i]
        model.model.train()
        opt.zero_grad(set_to_none=True)
        actions,logits=scoring(model,row["prompt"],row["original_action_order"])
        forward_counts["train_semantic_candidate_forwards"]+=len(actions)
        lp=torch.log_softmax(logits.float(),dim=0)
        chosen=actions.index(row["chosen_action"])
        rejected=actions.index(row["rejected_action"])
        if args.arm=="aware_dpo":
            margin=(lp[chosen]-lp[rejected])-refs[row["comparison_id"]]
            loss=-F.logsigmoid(args.beta*margin)
        else:
            margin=None
            loss=-lp[chosen]
        assert torch.isfinite(loss)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.0,
              error_if_nonfinite=True)
        assert torch.isfinite(grad)
        opt.step()
        curve.append({"step":step+1,"comparison_id":row["comparison_id"],
          "root":row["root"],"condition":row["condition"],
          "training_semantic_action":row["chosen_action"],
          "loss":float(loss.detach()),"reference_delta":float(margin.detach()) if margin is not None else None,
          "grad_preclip":float(grad),"sampled_source_state_count":1})
        if step==0 or (step+1)%12==0:
            print("V4_SEMANTIC_TRAIN",args.arm,step+1,
              "loss",round(float(loss.detach()),6),"grad",round(float(grad),4),
              "reference_delta",round(float(margin.detach()),5) if margin is not None else "CE",flush=True)
    args.output.mkdir(parents=True)
    model.save_adapter(args.output/"adapter")
    result={
       "experiment_cell":"V4_position_invariant_semantic_release",
       "method":args.arm,"base":args.base,"model_init":"fresh_seed7_base_NF4_BF16_LoRA_r8_a16",
       "optimizer":"AdamW","seed":args.seed,"steps":args.steps,
       "lr":args.lr,"beta":args.beta if args.arm=="aware_dpo" else None,
       "gradient_checkpointing":True,"max_prompt_tokens":4096,
       "lora_rank":8,"lora_alpha":16,"candidate_score_protocol":FORMAT,
       "one_shared_adapter_no_role_routing":True,
       "reference_for_DPO":"frozen_same_pretrained_base_no_adapter" if args.arm=="aware_dpo" else None,
       "history_blind_teacher_only_not_actor_observation":args.arm=="blind_ce",
       "train_input_sha256":sha(args.train),"n_independent_train_states":len(rows),
       "n_original_V4_option_records":2*len(rows),
       "n_generated_extra_option_swaps":0,
       "counterbalanced_semantic_candidate_prompts_identical":True,
       "teacher_label_distribution":dict(Counter(r["chosen_action"] for r in rows)),
       "training_root_ids":sorted({r["root"] for r in rows}),
       "sampling_schedule_sha256":sha256(json.dumps(idx,separators=(',',':')).encode()).hexdigest(),
       "sample_counts_by_group":dict(Counter(rows[i]["comparison_id"] for i in idx)),
       "forward_counts":dict(forward_counts),
       "DPO_reference_candidate_forwards":sum(len(r["canonical_actions"]) for r in rows) if args.arm=="aware_dpo" else 0,
       "wall_train_seconds":round(time.monotonic()-t,2),
       "native_Qstar_not_LLM_Qpi":True,"never_used_DG36_final_test":True,
       "curve":curve}
    (args.output/"TRAINING.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V4_SEMANTIC_FINISH",args.arm,"updates",args.steps,"seconds",result["wall_train_seconds"],flush=True)
if __name__=="__main__":train()
