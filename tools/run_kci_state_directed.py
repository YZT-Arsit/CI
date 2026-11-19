"""KCI-Align: state-directed native Agent rollout and matched PCCE pilot.

Use ONLY frozen source-adapted TRAIN state prefixes. This tool never writes
hidden simulator state, never changes the original terminal GRPO baseline and
never turns action oracle Qstar into an LM prompt.

Run:
  python tools/run_kci_state_directed.py probe --backend random --selection DIR --output DIR
  python tools/run_kci_state_directed.py probe --backend hf --model LOCAL --bits4 ... --probe-credit ...
  python tools/run_kci_state_directed.py train --backend hf --credit terminal|policy_cce
     --confirm-gpu-training --selection DIR --output DIR

CLI output is immutable, and every restart group retains its original source
scenario, native state hash, role, exact prior action prefix and sampled seeds.
"""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import argparse,json,random,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.core import UniformPolicy,aggregate,group_signal
from kci.agent_rl.state_restarts import RestartSpec,restore,sample_restart_group
from experiments.receipt_dataset import SOURCE

def hash_file(path):
    h=sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
def write(path,data):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def append(path,data):
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(data,ensure_ascii=False,sort_keys=True)+"\n")
        f.flush()
def records(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
def source_hashes(root):
    from kci.agent_rl.core import Scenario
    values={}
    for kind in ("meeting","query"):
        env=Scenario(kind,False,False,11).build(root)
        for key,h in env.source_hashes.items():
            if key in values and values[key]!=h:raise RuntimeError("Source hash conflict")
            values[key]=h
    return values
def load_selection(path,root):
    path=path.resolve()
    plan=json.loads((path/"SELECTION.json").read_text(encoding="utf-8"))
    raw=records(path/"FORKS.jsonl")
    if not 2<=len(raw)<=32 or len(raw)!=plan["selected_forks"]:
        raise ValueError("Bad frozen restart selection count")
    specs=[RestartSpec.from_record(r) for r in raw]
    if len({s.key for s in specs})!=len(specs):
        raise ValueError("Duplicate restart initial state")
    paired={}
    for i,s in enumerate(specs):
        # The group-pair contract prevents same-root history comparison leakage.
        paired.setdefault(s.pair_id,[]).append(s)
        restore(s,root)
    if any(len(group)!=2 or {s.scenario.related for s in group}!={False,True}
           or len({(s.scenario.kind,s.scenario.allow,s.scenario.horizon,
                    s.prefix,s.role,s.phase) for s in group})!=1
           for group in paired.values()):
        raise RuntimeError("Broken related/unrelated state-pair contract")
    if len(paired)*2!=len(specs):
        raise ValueError("Malformed paired state corpus")
    return specs,plan,{
       "FORKS.jsonl":hash_file(path/"FORKS.jsonl"),
       "SELECTION.json":hash_file(path/"SELECTION.json")}

def create_policy(a,trainable=False):
    if a.backend=="random":
        if trainable:raise ValueError("Random policy cannot be trained")
        return UniformPolicy()
    if not a.model:raise ValueError("HuggingFace model is required")
    from kci.agent_rl.hf_policy import HFActionPolicy
    return HFActionPolicy(model_path=a.model,
       trainable=trainable,quantization="4bit" if a.bits4 else "none",
       dtype=a.dtype,allow_download=False,
       max_prompt_tokens=a.max_prompt_tokens,
       temperature=a.temperature,explore_epsilon=a.explore_epsilon,
       lora_rank=a.lora_rank,lora_alpha=a.lora_alpha,
       gradient_checkpointing=a.gradient_checkpointing)

def evidence_config(a,selection,selection_hashes,root):
    revision=subprocess.run(
        ["git","-C",str(ROOT),"rev-parse","HEAD"],
        capture_output=True,text=True,check=False).stdout.strip()
    code_dir=ROOT/"code"/"kci"/"agent_rl"
    return {
       **vars(a),
       "selection_sha256":selection_hashes,
       "source_hashes":source_hashes(root),
       "code_sha256":{x.name:hash_file(x) for x in sorted(code_dir.glob("*.py"))},
       "runner_sha256":hash_file(Path(__file__).resolve()),
       "native_replay_auditor_sha256":hash_file(ROOT/"tools"/"audit_kci_state_restart.py"),
       "git_commit":revision,
       "selection_is_only_TRAIN_pinned_prefixes":True,
       "not_independent_business_TEST":True,
       "no_model_Qstar_in_policy_observations":True,
       "root_horizon_DEV_still_developed":True}

def sample_batch(spec,policy,a,index):
    return sample_restart_group(spec,policy,a.group_size,
       a.seed+1000000+2897*index,
       split="train",source_root=Path(a.source_root).resolve())

def score_group(group,spec,index,extra=None):
    s={"group_index":index,
       "group_id":spec.scenario.key,
       "restart":spec.serialize(),
       "sample_size":len(group),
       "rewards":[x.reward for x in group],
       "outcome_signal":group_signal(group),
       "reward_signal_variance":len({x.reward for x in group})>1,
       "safe_success":sum(e.success and not e.violation for e in group),
       "ci_violations":sum(e.violation for e in group),
       "safe_incomplete":sum(not e.success and not e.violation for e in group),
       "model_decisions":sum(len(e.decisions) for e in group),
       "first_action_counts":dict(__import__("collections").Counter(
           e.decisions[0].chosen_action for e in group)),
       "source_state_hash":spec.state_hash,
       "policy_id":group[0].policy_id}
    if extra:s.update(extra)
    return s

def probe_or_train(a,model,folder,specs):
    from run_kci_agent_rl import evaluate
    from kci.agent_rl.state_directed_cce import StateDirectedPCCE
    from kci.agent_rl.grpo import GRPOConfig,GRPOUpdater
    from kci.agent_rl.core import Scenario

    source_root=Path(a.source_root).resolve()
    groups=[]
    stats=[]
    extra_proofs=0
    supported_forks=0
    counterfactual_rollouts=0
    updater=None
    if a.mode=="train":
        cfg=GRPOConfig(lr=a.lr,clip_eps=a.clip_eps,
                       kl_coef=a.kl_coef,entropy_coef=0.,
                       grad_clip=1.,train_cost_only=False)
        updater=GRPOUpdater(model,cfg)
        write(folder/"GRPO_CONFIG.json",asdict(cfg))
    if a.credit_mode=="policy_cce" or a.probe_credit or a.matched_budget_teacher:
        teacher=StateDirectedPCCE(
           source_root=source_root,n_continuations=a.cce_rollouts,
           minimum_gap=a.cce_min_gap,
           minimum_consensus=a.cce_min_consensus,seed=a.seed+870112)
    else:teacher=None
    if teacher:
        write(folder/"STATE_CCE_CONFIG.json",{
           "teacher":"frozen_same_step_policy_Qpi_only",
           "restarts_are_real_immutable_TRAIN_native_prefixes":True,
           "K":a.cce_rollouts,
           "min_return_gap":a.cce_min_gap,
           "min_paired_direction_consensus":a.cce_min_consensus,
           "terminal_gradient_preserved_by_additive_credit":True,
           "local_weight":a.cce_weight,
           "no_ideal_Qstar_labels_in_credit":True,
           "no_curriculum_label_in_policy_prompt":True,
           "local_credit_applied_to_gradient":a.credit_mode=="policy_cce",
           "counterfactual_budget_only_control":bool(a.matched_budget_teacher)})
    dev_seed=a.seed+700000
    if a.mode=="train":
        ev=folder/"dev_before";ev.mkdir()
        before=evaluate(model,ev,"dev",dev_seed,a.dev_world_limit,a.group_size,source_root)
        write(ev/"SUMMARY.json",before)
    else:before=None

    for index,spec in enumerate(specs[:a.max_groups]):
        if updater:
            model.policy_id=f"hf:{a.model}:onpolicy_step_{updater.optimizer_steps}"
        group=sample_batch(spec,model,a,index)
        if teacher:
            credits,proofs,evidence=teacher.make_credits(
               spec,group,model,group_index=index)
            for proof in proofs:
                append(folder/"restart_cce_proofs.jsonl",proof)
            extra_proofs+=len(proofs)
            supported_forks+=evidence.get("supported_prefixes",0)
            counterfactual_rollouts+=evidence.get("policy_branch_rollouts",0)
        else:credits,proofs,evidence=None,[],{}
        if updater:
            if a.credit_mode=="policy_cce":
                result=updater.update(
                    group,local_credits=credits,cce_weight=a.cce_weight,
                    credit_style="additive")
            else:
                result=updater.update(group)
        else:result={"updated":False,"optimizer_step":0}
        record=score_group(group,spec,index,{
            **result,"state_cce_evidence":evidence,
            "restart_cce_proof_count":len(proofs)})
        for episode in group:
            append(folder/"episodes.jsonl",episode.serialize())
        append(folder/"groups.jsonl",record)
        groups.append(group)
        stats.append(record)
        print(json.dumps({
           "group":index+1,"of":a.max_groups,
           "restart_role":spec.role,
           "history_related":spec.scenario.related,
           "safe_success":record["safe_success"],
           "violations":record["ci_violations"],
           "outcome_signal":record["outcome_signal"],
           "supported_local_state":evidence.get("supported_prefixes",0),
           "local_credits":len(proofs),
           "updated":result["updated"],
           "optimizer_step":result["optimizer_step"]
        }),flush=True)
    if updater:
        model.save_adapter(folder/"adapter")
        model.policy_id=f"hf:{a.model}:after_{updater.optimizer_steps}"
        after_dir=folder/"dev_after";after_dir.mkdir()
        after=evaluate(model,after_dir,"dev",dev_seed,a.dev_world_limit,a.group_size,source_root)
        write(after_dir/"SUMMARY.json",after)
    else:after=None
    outcome=aggregate(groups)
    outcome.update({
        "mode":a.mode,"credit_mode":a.credit_mode,
        "restart_selection_groups":a.max_groups,
        "unique_restarted_state_hashes":len({x["restart"]["state_hash"] for x in stats}),
        "supported_counterfactual_initial_states":supported_forks,
        "accepted_local_credit_decisions":extra_proofs,
        "extra_policy_continuation_rollouts":counterfactual_rollouts,
        "model_trainable_gradient_steps":updater.optimizer_steps if updater else 0,
        "dev_before":before,"dev_after":after,
        "model_learning_gain_established":False,
        "test_family_independent":False,
        "matched_token_cost_with_root_GRPO":False,
    })
    write(folder/"SUMMARY.json",outcome)
    return outcome

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("probe","train"))
    p.add_argument("--selection",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--source-root",default=str(SOURCE))
    p.add_argument("--backend",choices=("random","hf"),default="random")
    p.add_argument("--model",default=None)
    p.add_argument("--bits4",action="store_true")
    p.add_argument("--dtype",choices=("bf16","fp16","fp32"),default="bf16")
    p.add_argument("--max-prompt-tokens",type=int,default=4096)
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--explore-epsilon",type=float,default=.35)
    p.add_argument("--group-size",type=int,default=8)
    p.add_argument("--max-groups",type=int,default=16)
    p.add_argument("--dev-world-limit",type=int,default=12)
    p.add_argument("--seed",type=int,default=9111926)
    p.add_argument("--credit-mode",choices=("terminal","policy_cce"),default="terminal")
    p.add_argument("--matched-budget-teacher",action="store_true",
                   help="Control arm computes the same Qpi branches but applies NO local credit")
    p.add_argument("--probe-credit",action="store_true",
                   help="Do frozen-policy Qpi evidence in a PROBE; no optimizer")
    p.add_argument("--cce-rollouts",type=int,default=4)
    p.add_argument("--cce-min-gap",type=float,default=.5)
    p.add_argument("--cce-min-consensus",type=float,default=.75)
    p.add_argument("--cce-weight",type=float,default=.5)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--clip-eps",type=float,default=.2)
    p.add_argument("--kl-coef",type=float,default=.02)
    p.add_argument("--lora-rank",type=int,default=8)
    p.add_argument("--lora-alpha",type=int,default=16)
    p.add_argument("--gradient-checkpointing",action="store_true")
    p.add_argument("--confirm-gpu-training",action="store_true")
    p.add_argument("--allow-cpu-smoke",action="store_true")
    return p.parse_args()

def main():
    a=parser()
    if not 2<=a.group_size<=32 or not 1<=a.max_groups<=32:
        raise ValueError("Invalid bounded rollout group count")
    if not 1<=a.dev_world_limit<=32:
        raise ValueError("Invalid DEV condition count")
    if not 0<=a.cce_weight<=1:
        raise ValueError("Invalid CCE weight")
    if not 2<=a.cce_rollouts<=16 or not 0<a.cce_min_gap<=2 or not .5<a.cce_min_consensus<=1:
        raise ValueError("Invalid Qpi confidence gate")
    if a.mode=="train":
        if a.backend!="hf" or not a.confirm_gpu_training:
            raise ValueError("Real model updates require HF and explicit --confirm-gpu-training")
        import torch
        if not torch.cuda.is_available() and not(
            a.allow_cpu_smoke and a.dtype=="fp32" and not a.bits4):
            raise RuntimeError("No CUDA GPU; 4B CPU training is forbidden")
        random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    if a.mode=="probe" and a.credit_mode=="policy_cce":
        raise ValueError("Probe uses --probe-credit; no training credit labels")
    if a.mode=="train" and a.probe_credit:
        raise ValueError("Training uses --credit-mode policy_cce")
    if a.matched_budget_teacher and (a.mode!="train" or a.credit_mode!="terminal"):
        raise ValueError("Matched-budget teacher can be used only with terminal-only TRAIN control")
    root=Path(a.source_root).resolve()
    path=Path(a.selection).resolve()
    specs,selected,hashes=load_selection(path,root)
    if a.max_groups>len(specs):raise ValueError("More groups than frozen restart selection")
    model=create_policy(a,trainable=a.mode=="train")
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("No overwrites: immutable output directory exists")
    out.mkdir(parents=True,exist_ok=True)
    write(out/"RUN_CONFIG.json",evidence_config(a,selected,hashes,root))
    result=probe_or_train(a,model,out,specs)
    manifest={x.relative_to(out).as_posix():hash_file(x)
              for x in out.rglob("*") if x.is_file()}
    write(out/"MANIFEST.json",manifest)
    print(json.dumps({"status":"COMPLETE","output":str(out),
         "groups":a.max_groups,
         "optimizer_steps":result["model_trainable_gradient_steps"],
         "supported_restart_states":result["supported_counterfactual_initial_states"],
         "extra_counterfactual_policy_rollouts":result["extra_policy_continuation_rollouts"]}),flush=True)
if __name__=="__main__":main()
