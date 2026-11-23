"""Freeze Phase2 history-contrast controlled state pool, reused for all arms.

No invented contexts/labels. Existing Phase2 frozen role-visible observations,
native executed RAW/VERIFY action outcomes and 4 context relation witnesses.
Three arms see exactly the same five real release states per TRAIN root.
DEV is separately written, never opened by the trainers. Final sealed tests
never deserialized. Retains known tight-deadline no-success risk cases.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
SOURCE=Path(r"D:\research\kci_data_v4_phase2_20261122\artifacts\20261122_kci_data_v4_phase2_full_v1")
OUT=ROOT/"artifacts/20261124_joint_history_supervision_v1"
TRAIN=("P201","P202","P203","P205","P206","P207","P208")
DEV=("P204","P209","P210")
ALL=TRAIN+DEV
CASES=("risky","safe","hold","authorized","equivalent")
RELATIONS={"history_flip":("safe","risky"),"unrelated_hold":("safe","hold"),
           "authorization_exception":("risky","authorized"),
           "equivalent_history":("safe","equivalent")}
RAW="SEND_RAW_TARGET";VERIFY="VERIFY_PRIVATE_TARGET";ABORT="ABORT"
def jl(p):
 rows=[]
 with p.open(encoding="utf8") as f:
  for line in f:
   if not any(('"root_id":"'+r+'"') in line for r in ALL):continue
   if '"final_test"' in line or '"P211' in line or '"P212' in line:
    raise AssertionError("sealed source crossing train/dev boundary")
   item=json.loads(line)
   assert item["root_id"] in ALL and item["split"] in ("train","dev")
   rows.append(item)
 return rows
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 x=jl(SOURCE/"MODEL_ROLE_OBSERVATIONS_EVAL_ONLY.jsonl")
 futures=jl(SOURCE/"FULL_NATIVE_ACTION_TRAJECTORIES_EVAL_ONLY.jsonl")
 pairs=jl(SOURCE/"CONTROLLED_CONTEXT_PAIRS_EVAL_ONLY.jsonl")
 assert len(x)==len(futures)==50 and len(pairs)==40
 xf={r["world_id"]:r for r in x}
 tf={r["world_id"]:r for r in futures}
 assert set(xf)==set(tf)
 pc=defaultdict(dict)
 for p in pairs:
  root=p["root_id"];category=p["kind"]
  assert category in RELATIONS and category not in pc[root]
  assert p["same_current_signed_source"] and p["same_history_size"] and p["same_task_and_role"]
  assert p["not_same_state_DPO"] and p["same_sent_raw_action"]
  left,right=RELATIONS[category]
  assert (p["left_world"],p["right_world"])==(root+"__"+left,root+"__"+right),p
  pc[root][category]=p
 rows=[]
 for root in ALL:
  assert set(pc[root])==set(RELATIONS)
  for cond in CASES:
   world=root+"__"+cond
   sample=xf[world];truth=tf[world]
   assert sample["role_observation"]["stage"]=="release"
   assert sample["role_observation"]["actor"]=="SourceAgent"
   assert sample["model_only_legally_observed_data"] and sample["model_no_future_oracle_labels"]
   from kci.diverse_ci_branch_env import digest
   assert sample["native_state_sha256"]==digest(truth["fully_replayable_native_decision_state"])
   # Check native state actually contains same role-legally observed ledger and pending source.
   native=truth["fully_replayable_native_decision_state"]
   assert native["stage"]=="release" and not native["done"]
   assert sample["role_observation"]["signed_source_tool_record"]==native["source_receipt"]
   target=sample["role_observation"]["recipient"]
   assert sample["role_observation"]["delivered_message_log"]==[
      e for e in native["ledger"] if e["recipient"]==target]
   branches=truth["native_action_branches"]
   assert RAW in branches and VERIFY in branches
   r=branches[RAW];v=branches[VERIFY]
   assert r["immediate_verdict"]==("VIOLATION" if cond=="risky" else "VALID"),(world,r["immediate_verdict"])
   assert v["immediate_verdict"]=="VALID"
   # Preserve CI-first source action teacher, with actual safe-success/cost
   # separate. If risky VERIFY times out, this DOES NOT create a success claim.
   gold=VERIFY if cond=="risky" else RAW
   assert (gold==RAW)==(r["immediate_verdict"]=="VALID")
   rows.append({
     "root":root,"world":world,"condition":cond,"split":sample["split"],
     "stage":"release","role":"SourceAgent",
     "actor_visible_observation":sample["role_observation"],
     "actor_observation_sha256":sample["role_observation_sha256"],
     "native_state_sha256":sample["native_state_sha256"],
     "role_visible_text":sample["model_text"],
     "entity_alias_text":sample["entity_renamed_text"],
     "option_swap_text":sample["option_swapped_text"],
     "available_native_actions":sample["role_observation"]["legal_actions"],
     "native_gold":gold,"same_state_rejected":RAW if gold==VERIFY else VERIFY,
     "native_immediate_verdict":{a:branches[a]["immediate_verdict"] for a in (RAW,VERIFY)},
     "fixed_native_suffix":{a:{
        "success":bool(branches[a]["real_fixed_suffix"]["success"]),
        "violation":bool(branches[a]["real_fixed_suffix"]["violation"]),
        "deadline":bool(branches[a]["real_fixed_suffix"]["deadline_missed"]),
        "total_steps":int(branches[a]["real_fixed_suffix"]["total_steps"]),
        "grade":"native_fixed_policy_witness_not_Qstar_not_Qpi"} for a in (RAW,VERIFY)},
     "verified_native_source_state":native,
     "source_evidence_grade":"real_native_immediate_verdict_with_fixed_suffix_only",
     "not_a_CCE_longterm_teacher":True})
 gr=defaultdict(dict)
 for r in rows:gr[r["root"]][r["condition"]]=r
 for root,group in gr.items():
  assert set(group)==set(CASES)
  assert all(x["split"]==group["safe"]["split"] for x in group.values())
  assert all(x["native_gold"]==("VERIFY_PRIVATE_TARGET" if x["condition"]=="risky" else "SEND_RAW_TARGET") for x in group.values())
  assert len({x["actor_observation_sha256"] for x in group.values()})==5
  # Policy authorization is stored in the environment contract, not its FlowState;
  # two equal FlowState hashes alone never imply equal authorization.
  for typ,(left,right) in RELATIONS.items():
   assert pc[root][typ]["left_world"]==group[left]["world"]
   assert pc[root][typ]["right_world"]==group[right]["world"]
 assert len(rows)==50
 OUT.mkdir(parents=True)
 for split,roots in (("TRAIN",TRAIN),("DEV",DEV)):
  group=[r for r in rows if r["root"] in roots]
  p=OUT/(split+"_FIVE_CONTEXTS.jsonl")
  p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in group),encoding="utf8")
 for split,roots in (("TRAIN",TRAIN),("DEV",DEV)):
  group=[p for p in pairs if p["root_id"] in roots]
  (OUT/(split+"_RELATION_PAIRS.jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in group),encoding="utf8")
 train=OUT/"TRAIN_FIVE_CONTEXTS.jsonl";dev=OUT/"DEV_FIVE_CONTEXTS.jsonl"
 manifest={
  "cell":"matched_three_arm_existing_true_history_continuation",
  "not_new_algorithm":"extends existing tools/kci_v4_history_contrast_train.py",
  "train_roots":list(TRAIN),"dev_roots":list(DEV),
  "n_train_roots":len(TRAIN),"n_dev_roots":len(DEV),
  "n_train_true_states":len(TRAIN)*5,"n_dev_true_states":len(DEV)*5,
  "five_existing_conditions":list(CASES),"pair_types":list(RELATIONS),
  "all_three_arms_have_same_35_actor_visible_states":True,
  "same_state_DPO_only_not_cross_history_DPO":True,
  "risky_VERIFY_may_fail_tight_deadline_and_is_not_claimed_safe_success":True,
  "native_fixed_suffix_not_true_Qpi":True,
  "final_test_never_parsed":True,
  "no_CCE_reward_or_Qstar_supervision":True,
  "source_full_native_trajectories_sha256":sha(SOURCE/"FULL_NATIVE_ACTION_TRAJECTORIES_EVAL_ONLY.jsonl"),
  "source_role_observations_sha256":sha(SOURCE/"MODEL_ROLE_OBSERVATIONS_EVAL_ONLY.jsonl"),
  "source_pair_evidence_sha256":sha(SOURCE/"CONTROLLED_CONTEXT_PAIRS_EVAL_ONLY.jsonl"),
  "train_frozen_sha256":sha(train),"dev_frozen_sha256":sha(dev),
  "train_pairs_sha256":sha(OUT/"TRAIN_RELATION_PAIRS.jsonl"),
  "dev_pairs_sha256":sha(OUT/"DEV_RELATION_PAIRS.jsonl")}
 (OUT/"FROZEN_CONTRACT.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 print("MATCHED_HISTORY_DATA_FROZEN",json.dumps(manifest,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
