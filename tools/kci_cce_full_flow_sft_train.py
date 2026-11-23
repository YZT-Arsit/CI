"""Phase2 shared 4B native successful-workflow Action SFT.

Invariants:
- One Qwen3-4B base + ONE fresh LoRA across ALL roles.
- Native ABORT remains an executable model choice; no reward changed.
- Only verified complete successful TRAIN native witness observations.
- No A_SEND_PACKAGE_A/B targets: upstream long-horizon A remains untrained
  for future Action CE / terminal / CCE preference comparisons.
- Root + stage balanced, fixed one-shot regimen; NO dev or final-test input.
This is a capability baseline, not CCE/DPR research contribution.
"""
from __future__ import annotations
from collections import Counter,defaultdict,deque
from hashlib import sha256
from pathlib import Path
import argparse,json,random,sys,time,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci.agent_rl.hf_policy import HFActionPolicy
from kci_cce_phase2_frozen4b_rollout import model_candidate_prompt
from torch.nn.functional import log_softmax
STAGE_CYCLE=(
    "pre","pre","preflight","preflight","evidence","evidence",
    "release","release","release","attest",
    "post","post","broker","A_READ")
ROOTS=("P201","P202","P203","P205","P206","P207","P208")
RUN_STEPS=160
BATCH=2
SEED=7
LR=2e-5

def sha(p):
    h=sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()
def load(path):
    rows=[json.loads(line) for line in Path(path).open(encoding="utf8") if line.strip()]
    assert len(rows)>500
    assert {r["root"] for r in rows}==set(ROOTS)
    assert len({r["same_observation_sha256"] for r in rows})==len(rows)
    assert all(r["split"]=="train" and r["verified_native_full_success_witness"] for r in rows)
    assert all("ABORT" in r["all_executable_actions"] for r in rows)
    assert all(r["chosen_action"]!="ABORT" for r in rows)
    assert all(r["stage"]!="A_SEND" and r["not_a_future_A_preference"] for r in rows)
    assert all(r["native_role_observation"]["actor"]==r["role"] for r in rows)
    assert all(set(r["all_executable_actions"])==set(r["native_role_observation"]["legal_actions"]) for r in rows)
    assert all("Qstar" not in r["native_role_observation"] for r in rows)
    return rows

def balanced_schedule(rows,seed=SEED,steps=RUN_STEPS,batch=BATCH):
    random_rng=random.Random(seed)
    buckets=defaultdict(lambda:defaultdict(list))
    for row in rows:buckets[row["stage"]][row["root"]].append(row)
    assert set(STAGE_CYCLE)==set(buckets)
    ptr=Counter()
    for stage,roots in buckets.items():
        for root,sub in roots.items():
            sub.sort(key=lambda x:x["same_observation_sha256"])
            random_rng.shuffle(sub)
    schedule=[]
    n=steps*batch
    for i in range(n):
        stage=STAGE_CYCLE[i%len(STAGE_CYCLE)]
        # Every stage cycles through available task roots, not row frequency.
        roots=sorted(buckets[stage])
        root=roots[ptr[("stage",stage)]%len(roots)]
        ptr[("stage",stage)]+=1
        pool=buckets[stage][root]
        chosen=pool[ptr[("root",stage,root)]%len(pool)]
        ptr[("root",stage,root)]+=1
        schedule.append(chosen)
    assert len(schedule)==n
    assert set(x["stage"] for x in schedule)==set(STAGE_CYCLE)
    assert set(x["root"] for x in schedule)==set(ROOTS)
    return schedule

def candidate_scores(hf,obs):
    actions=tuple(sorted(obs["legal_actions"]))
    assert 2<=len(actions)<=5
    results=[]
    for act in actions:
        prompt=model_candidate_prompt(obs,act)
        logp=hf.log_distribution(prompt,("ACCEPT","REJECT"))
        results.append(logp[0]-logp[1])
    return actions,torch.stack(results)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--base",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=RUN_STEPS)
    p.add_argument("--batch",type=int,default=BATCH)
    p.add_argument("--lr",type=float,default=LR)
    p.add_argument("--seed",type=int,default=SEED)
    args=p.parse_args()
    assert (args.steps,args.batch,args.lr,args.seed)==(RUN_STEPS,BATCH,LR,SEED)
    if args.output.exists():raise FileExistsError(args.output)
    rows=load(args.train)
    schedule=balanced_schedule(rows,args.seed,args.steps,args.batch)
    random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    hf=HFActionPolicy(model_path=args.base,trainable=True,
        quantization="4bit",dtype="bf16",max_prompt_tokens=4096,
        lora_rank=8,lora_alpha=16,gradient_checkpointing=True,
        allow_download=False,temperature=1.,explore_epsilon=0.)
    assert "fresh-lora" in hf.policy_id
    opt=torch.optim.AdamW(hf.trainable_parameters(),lr=args.lr,weight_decay=0.)
    times=time.monotonic();curve=[];total_forwards=0;loss_by_stage=defaultdict(list)
    print("CCE_FLOW_SFT_TRAIN_START","native_train_roots",len(ROOTS),
          "unique_train_states",len(rows),"steps",args.steps,"batch",args.batch,
          "stage_cycle",STAGE_CYCLE,flush=True)
    for step in range(args.steps):
        hf.model.train()
        opt.zero_grad(set_to_none=True)
        hits=0;items=[]
        for k in range(args.batch):
            row=schedule[step*args.batch+k]
            actions,scores=candidate_scores(hf,row["native_role_observation"])
            assert "ABORT" in actions and row["chosen_action"] in actions
            logprobs=log_softmax(scores.float(),dim=0)
            gold=actions.index(row["chosen_action"])
            loss=-logprobs[gold]
            assert torch.isfinite(loss)
            (loss/args.batch).backward()
            hits+=int(actions[int(scores.detach().argmax())]==row["chosen_action"])
            total_forwards+=len(actions)
            value=float(loss.detach())
            loss_by_stage[row["stage"]].append(value)
            items.append({"root":row["root"],"stage":row["stage"],"role":row["role"],
                "label":row["chosen_action"],"loss":value,
                "n_native_candidates":len(actions),"has_ABORT":True})
        grad=torch.nn.utils.clip_grad_norm_(hf.trainable_parameters(),1.,
                                           error_if_nonfinite=True)
        assert torch.isfinite(grad)
        opt.step()
        curve.append({"step":step+1,"samples":items,
            "mean_loss":sum(x["loss"] for x in items)/len(items),
            "train_hits_preupdate":hits,"grad_preclip":float(grad)})
        if step==0 or (step+1)%16==0:
            print("CCE_FLOW_SFT_TRAIN",step+1,"loss",round(curve[-1]["mean_loss"],4),
                  "training_batch_hits",hits,"grad",round(float(grad),4),
                  "stages",[(x["stage"],x["role"]) for x in items],flush=True)
    args.output.mkdir(parents=True)
    hf.save_adapter(args.output/"adapter")
    meta={
      "cell":"KCI_CCE_downstream_execution_capacity_SHARED_Action_SFT",
      "method":"single_shared_native_success_witness_Action_CE_SFT",
      "research_role":"downstream_executor_capability_baseline_not_CCE_algorithm",
      "source_verified_native_witness_train_sha256":sha(args.train),
      "train_unique_actor_states":len(rows),
      "sampled_sft_steps":args.steps,"sampled_train_observation_updates":args.steps*args.batch,
      "batch":args.batch,"seed":args.seed,"lr":args.lr,"optimizer":"AdamW",
      "base":args.base,"model_init":"new_single_LoRA_from_same_Qwen3_4B_base",
      "quantization":"NF4_BF16","lora_rank":8,"lora_alpha":16,
      "lora_target_modules":["q_proj","v_proj"],"gradient_checkpointing":True,
      "one_shared_adapter_for_all_roles":True,
      "role_specific_adapter_routing":False,
      "ABORT_preserved_in_every_model_menu":True,
      "native_rewards_and_verifier_unchanged":True,
      "A_SEND_not_supervised_by_success_witnesses":True,
      "future_CCE_preference_not_used_as_supervisor":True,
      "oracle_Qstar_not_used_as_current_4B_Qpi":True,
      "TRAIN_roots":list(ROOTS),"DEV_and_final_roots_never_read":True,
      "sampling_scheme":"deterministic_stage_cycle_root_round_robin_witnesses",
      "stage_cycle":list(STAGE_CYCLE),
      "sample_count_by_stage":dict(Counter(x["stage"] for x in schedule)),
      "sample_count_by_root":dict(Counter(x["root"] for x in schedule)),
      "sample_count_by_action":dict(Counter(x["chosen_action"] for x in schedule)),
      "schedule_sha256":sha256(json.dumps([x["same_observation_sha256"] for x in schedule],separators=(",",":")).encode()).hexdigest(),
      "differentiable_candidate_forwards":total_forwards,
      "training_wall_seconds_excludes_model_loading":round(time.monotonic()-times,2),
      "stage_loss_averages":{k:sum(v)/len(v) for k,v in loss_by_stage.items()},
      "curve":curve,
    }
    (args.output/"TRAINING.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CCE_FLOW_SFT_TRAIN_COMPLETE",args.steps,total_forwards,
          "wall_s",meta["training_wall_seconds_excludes_model_loading"],flush=True)
if __name__=="__main__":main()
