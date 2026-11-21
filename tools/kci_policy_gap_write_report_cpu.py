r"""Source-backed Chinese research report for EXACT frozen Qpi and 4 CPU methods.

Everything reads immutable saved native trajectories, Qstar certification,
checkpointed frozen-policy Qpi, fresh post-train Qpi and independent replay.
No new synthetic task, no fabricated method benefit, no LLM weight update.
"""
from pathlib import Path
from collections import defaultdict,Counter
from statistics import mean,stdev
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_exact_frozen_qpi_cpu import ROWS
from compare_kci_chain_methods_cpu_action_interaction import GROUP,build_expert_dataset
B=ROOT/"artifacts"/"20261121_policy_gap_pilot_01"
PRE=B/"SIX_SAVED_CHECKPOINTS_QSTAR_VS_EXACT_QPI"
TEACH=B/"TRAIN_ONLY_QPI_CREDIT_FROM_SFT_REFERENCE"
TRAIN=B/"FOUR_WAY_CPU_TRAINED_POLICY_RESULTS_03"
POST=B/"EXACT_QPI_BEFORE_AFTER_FOUR_METHODS_02"
AUDIT=B/"INDEPENDENT_EXHAUSTIVE_QPI_NATIVE_REPLAY"
INTER=B/"DG19_PRIVILEGED_CAUSAL_ROLE_INTERVENTION"
SOURCE=B/"DG19_SOURCE_ONLY_INTERVENTION_DEFINITION_CHECK"
OUT=B/"RESEARCH_REPORT"
if OUT.exists():raise FileExistsError(OUT)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
pre=load(PRE/"SUMMARY.json")
teacher=load(TEACH/"SUMMARY.json")
train=load(TRAIN/"TRAINING_SUMMARY.json")
post=load(POST/"SUMMARY.json")
ind=load(AUDIT/"SUMMARY.json")
a=load(INTER/"DG19_CAUSAL_INTERVENTIONS.json")
s=load(SOURCE/"SOURCE_INTERVENTIONS.json")
assert pre["status"]=="ALL_SIX_SAVED_CPU_CHECKPOINTS_EXACT_QPI_REPRODUCED"
assert teacher["status"]=="TRAIN_ONLY_EXACT_FROZEN_SFT_QPI_REFERENCE_CREDIT"
assert train["status"]=="FOUR_WAY_SHARED_CPU_AGENT_REAL_NATIVE_TRAINING_COMPLETE"
assert post["status"]=="ALL_FOUR_METHODS_EXACT_QPI_RECOMPUTED_BEFORE_AND_AFTER_NATIVE"
assert ind["status"]=="INDEPENDENT_FROZEN_AND_TRAINED_Qpi_NATIVE_DFS_REPLAY_PASS"
assert len(post["runs"])==12
methods=("action_sft","qstar_dpo","frozen_qpi_dpo","terminal_reinforce")
names={
  "action_sft":"Action SFT",
  "qstar_dpo":"Exact Q-star DPO",
  "frozen_qpi_dpo":"Frozen Q-pi DPO",
  "terminal_reinforce":"Terminal REINFORCE",
}
train_rows=[json.loads(x) for x in (TRAIN/"TRAINING_EVALUATIONS.jsonl").read_text(
  encoding="utf-8").splitlines() if x.strip()]
assert len(train_rows)==12
before_after={(x["method"],x["seed"]):x for x in post["runs"]}
trained={(x["method"],x["seed"]):x for x in train_rows}
assert len(before_after)==len(trained)==12
after_details=[json.loads(x) for x in (POST/"ALL_EXACT_PRE_POST_POLICY_REEVALUATIONS.jsonl").read_text(
  encoding="utf-8").splitlines() if x.strip() and '"policy_state":"after"' in x]
assert len(after_details)==12*28
details={(x["policy_kind"],x["seed"],x["world"]):x for x in after_details}
result={}
for m in methods:
  rr=[before_after[m,seed] for seed in (7,1234,2025)]
  ts=[trained[m,seed]["exploratory_test"]["actual_native_greedy"]
      for seed in (7,1234,2025)]
  exp=[x["means"]["after_exploratory_test_mean_Qpi_full_return"] for x in rr]
  leak=[x["means"]["after_exploratory_test_mean_CI_violation_probability"] for x in rr]
  conflicts=[]
  preconf=[]
  regret=[]
  for seed in (7,1234,2025):
    r=before_after[m,seed]
    conflicts.append(sum(r["counts"]["after_"+split+"_strict_Qstar_Qpi_conflicts"]
           for split in ("train","dev","exploratory_test")))
    preconf.append(sum(r["counts"]["before_"+split+"_strict_Qstar_Qpi_conflicts"]
           for split in ("train","dev","exploratory_test")))
    vals=[max(ROWS[w["world"]]["upstream_A_values_given_exact_B_continuation"].values())-
          details[m,seed,w["world"]]["full_current_policy_native_Qpi"]
          for w in GROUP["exploratory_test"]]
    assert all(v>=-1e-6 for v in vals)
    regret.append(mean(vals))
  result[m]={
    "safe_task_completion":[x["safe_success"] for x in ts],
    "native_CI_violation":[x["ci_violations"] for x in ts],
    "A_best_or_tied":[x["A_optimal_action_given_eval_truth"] for x in ts],
    "B_correct_on_actual_received_history":[x["B_correct_on_visited_history"] for x in ts],
    "greedy_native_mean_return":[x["mean_native_return"] for x in ts],
    "expected_Qpi_reward":exp,"expected_Qpi_reward_mean":mean(exp),
    "expected_Qpi_reward_sample_SD":stdev(exp),
    "expected_native_violation_prob":leak,
    "expected_native_violation_prob_mean":mean(leak),
    "Qstar_regret_expected_policy":regret,
    "Qstar_regret_expected_policy_mean":mean(regret),
    "strict_conflicts_before":preconf,"strict_conflicts_after":conflicts,
    "test_conflicts_after":[x["counts"]["after_exploratory_test_strict_Qstar_Qpi_conflicts"] for x in rr],
    "training_native_sampled_episodes":[trained[m,seed]["actual_training_cost"].get(
       "native_training_episodes",0) for seed in (7,1234,2025)]
  }
assert all(v==10 for v in pre["Qstar_Qpi_conflicts_per_checkpoint"].values())
source_only=next(x for x in s["results"] if x["mode"]==
    "source_OPTIMAL_among_RAW_VERIFY_ABORT_all_release_states")
source_effect=source_only["Qpi_A_minus_B_under_this_privileged_intervention"]
assert abs(source_effect-0.01308)<1e-4
assert abs(a["conditions"][1]["Qpi_A_minus_B_only_original_when_no_intervention"]-0.01762)<1e-4
assert abs(a["conditions"][3]["Qpi_A_minus_B_only_original_when_no_intervention"]-0.028)<1e-5
native_cost=build_expert_dataset(GROUP["train"])[3]["actual_native_reference_expansions"]

lines=[
"# KCI-Align：原生 Q-star 与冻结共享策略 Q-pi 的排序失配",
"",
"## 实验范围",
"",
"原28个真实可执行的合成 A→R→B→C 世界、7种知识图、固定 4/1/2 类知识图 TRAIN/DEV/探索测试划分。来源环境、任务标签和原模型检查点全部只读；没有新任务数据、没有GPU、没有训练Qwen模型。",
"",
"Q-pi 是由加载PyTorch共享CPU Agent检查点，在当前状态下使用合法动作softmax，完整枚举全部下游真实原生状态转移计算的策略期望终局价值。不会把原生Q-star标签注入执行策略。",
"",
"## 1. 六个冻结检查点：稳定冲突",
"",
"每个源世界A先从同一个已经签名的真实工具读取状态发出PACKAGE_A或PACKAGE_B；两条上游动作当前都合法，下游所有角色由冻结策略行动概率控制。六个检查点每次28世界均有24严格非平局、4个真实Q-star价值平局，其中10个严格Q-star/Q-pi排序相反。",
f"六个检查点累计验证{pre['total_native_states']}原生状态、{pre['total_native_edges']}实际工具/业务转移、{pre['total_policy_softmax_calls']}次冻结模型决策分布。已经过独立DFS逐分支重算。",
"",
"DG19_M01__tradeoff__swap0 的SFT seed7案例：",
"",
"| 续行或角色干预 | Q(A)-Q(B) |",
"|---|---:|",
"| Native optimal Q-star | +0.02800 |",
f"| **不干预冻结模型 Q-pi** | {a['conditions'][0]['Qpi_A_minus_B_only_original_when_no_intervention']:+.5f} |",
f"| 只把B的OPEN/SKIP纠正为真实native最佳 | {a['conditions'][1]['Qpi_A_minus_B_only_original_when_no_intervention']:+.5f} |",
f"| 只纠正来源Agent的RAW/VERIFY/ABORT | {source_effect:+.5f} |",
f"| 两处都纠正，C仍按原策略 | {a['conditions'][3]['Qpi_A_minus_B_only_original_when_no_intervention']:+.5f} |",
"",
"只纠正来源Agent的定义至关重要：未预先开通私人验证时仍应允许合法ABORT，不应强制错误RAW披露。只在RAW和VERIFY同时合法时纠正，数值不会得到+0.01308。特权干预只作因果检验，绝非Q-pi教师或训练标签。",
"",
"原冻结模型强制A发高风险包时真实泄漏概率为6.73%，另一包接近0。因此排序反转来自下游实际失败概率，而不是模拟器修改了A动作当前合法性。",
"",
"## 2. 训练图冻结Q-pi教师",
"",
"每个seed原SFT检查点在算教师期间完全冻结，源数据仅16个训练世界：16组上游A两行动和32组真实递交之后的Coordinator OPEN/SKIP对。共3seed各计算48组同状态行为比较，均保存完整model actor observation和checkpoint哈希。",
f"生成教师共访问{teacher['total_Qpi_native_states']}原生状态、{teacher['total_Qpi_native_actions']}真实边及{teacher['total_frozen_actor_softmax_calls']}次策略softmax。每seed训练图中A存在6组Q-star/Q-pi偏好冲突，B没有冲突。两个Q-star真平局训练A状态两种DPO都剔除，不用人为固定动作。",
"",
"## 3. 同起点、同120更新的四方法训练结果",
"",
"各seed四方法均加载同一个原SFT共享策略（4471参数），Adam学习率0.015，再进行120次更新，合法动作、业务工具、角色观测保持一致。普通Action SFT得到和DPO一样的反事实状态覆盖及正确动作示范。Q-star DPO采信原生最优行动价值；Q-pi DPO采信冻结SFT参考策略的真实续行。Terminal REINFORCE每次仅访问采样轨迹终局奖励。",
"",
"| 方法 | 贪心安全任务完成 (3 seeds，每个8) | 贪心泄漏(3 seeds) | 上游A最优/等价 | 随机策略期望Q-pi均值±样本SD | 随机策略泄漏概率均值 |",
"|---|---|---|---|---:|---:|",
]
for k in methods:
  x=result[k]
  lines.append("| "+names[k]+" | "+",".join(map(str,x["safe_task_completion"]))+
    " | "+",".join(map(str,x["native_CI_violation"]))+
    " | "+",".join(map(str,x["A_best_or_tied"]))+
    f" | {x['expected_Qpi_reward_mean']:.4f} ± {x['expected_Qpi_reward_sample_SD']:.4f} | {100*x['expected_native_violation_prob_mean']:.2f}% |")
lines += [
"",
"Q-pi DPO不是唯一能纠正下游策略的目标。本次同源CPU对照中，普通Action SFT的完整策略期望回报更好，不能宣称Q-pi DPO在学习效果上成立。策略执行的平均回报和绝对泄漏概率是原生真实事务与收件历史计算结果，不是静态标签评分。",
"",
"## 4. 每个训练后模型的Q-pi重新计算",
"",
"| 方法 | Q-star/Q-pi严格排序冲突：训练前3seed | 训练后3seed | 探索测试图训练后冲突 | 当前策略对Q-star最优的期望Regret均值 |",
"|---|---|---|---|---:|",
]
for k in methods:
  x=result[k]
  lines.append("| "+names[k]+" | "+",".join(map(str,x["strict_conflicts_before"]))+
    " | "+",".join(map(str,x["strict_conflicts_after"]))+
    " | "+",".join(map(str,x["test_conflicts_after"]))+
    f" | {x['Qstar_regret_expected_policy_mean']:.4f} |")
lines += [
"",
"以上Q-pi全部是重新加载更新后权重的原生精确策略期望，不是训练前冻结参考teacher的值。训练前Q-pi只能代表冻结参考策略；模型改变后，原有偏好可能改变方向。",
"",
"## 5. 监督成本",
"",
f"原生Q-star训练参考监督需约{native_cost}个native搜索状态；冻结Q-pi教师三个seed另需{teacher['total_Qpi_native_states']}原生访问状态。Terminal REINFORCE每seed120更新×16训练世界=1920条完整训练期原生轨迹。一次状态枚举与一次真实模型调用不能直接换算成本；训练更新次数一致不代表监督预算完全相等。",
"",
"训练前六个模型的完整Q-pi复核、训练后12个模型的重新Q-pi及独立DFS回放，属于实验审计成本，不可算作训练阶段已有的免费参考。",
"",
"## 6. 科研结论",
"",
"结论A：真实Q-star/Q-pi排序失配可在所有六个检查点稳定复现。DG19中纠正B和source_agent真实可改变排序，说明是下游策略性失误，而不是人工标签错误。",
"",
"结论B：冻结Q-pi偏好teacher没有让共享CPU策略优于拥有相同状态覆盖的Action SFT或Q-star DPO；在本轮探索图中SFT的期望回报/泄漏概率更好。原生最优teacher失配是真实的，但不是Q-pi DPO改进的充分条件。",
"",
"结论C：同一Q-pi教师固定训练可能鼓励当前策略的下游失败偏好，且当共享策略更新时Q-pi价值目标漂移。缺少可靠改进不应通过更改测试图、扩大随机超参搜索或修改任务奖励来掩盖。",
"",
"严格界限：这是一组有显式Horn归纳偏置的4471参数CPU共享策略、7类合成知识图、2个已参与开发的探索测试家族。不是独立盲测，也不是目标4B LLM、正式TRl GRPO/CCE训练，不构成直接ICML方法贡献。",
]
OUT.mkdir()
(OUT/"FINDINGS_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
payload={
 "status":"SOURCE_AUDITED_FROZEN_QSTAR_QPI_MISMATCH_FOURWAY_CPU_COMPARISON",
 "seeds":[7,1234,2025],
 "original_source_worlds":28,"source_data_modified":False,
 "six_frozen_checkpoints_strict_conflicts_each":10,
 "original_A_teacher_disagreement_train_each":6,
 "four_method_models_trained":12,
 "independent_native_DFS_replay_pass":True,
 "metrics":result,
 "not_Qwen_weights_updated":True,
 "no_blind_test":True,
 "source_sha256":{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
    for p in (PRE/"ALL_SIX_FROZEN_QPI_EVIDENCE.jsonl",
              TEACH/"SUMMARY.json",TRAIN/"TRAINING_EVALUATIONS.jsonl",
              POST/"ALL_EXACT_PRE_POST_POLICY_REEVALUATIONS.jsonl",
              AUDIT/"SUMMARY.json",INTER/"DG19_CAUSAL_INTERVENTIONS.json",
              SOURCE/"SOURCE_INTERVENTIONS.json")}
}
(OUT/"RESULTS.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("FINAL_FINDINGS",json.dumps({
  "status":payload["status"],
  "expected_return":{k:round(v["expected_Qpi_reward_mean"],6) for k,v in result.items()},
  "leak_prob":{k:round(v["expected_native_violation_prob_mean"],6) for k,v in result.items()},
  "post_conflicts":{k:v["strict_conflicts_after"] for k,v in result.items()},
},ensure_ascii=False),flush=True)
