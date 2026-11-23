"""Summarize the three independently reproducible CPU archive diagnostics."""
import json
import statistics
from pathlib import Path
from diagnose_state_matched_cpu import OUT,ARCHIVE

def load(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))

a=load("CPU_CREDIT_DIAGNOSTIC.json")
b=load("CPU_LOGPROB_DIAGNOSTIC.json")
c=load("CPU_HISTORY_PAIR_DIAGNOSTIC.json")
g=a["credit_mechanism"]["groups"]
metric={}
for arm in ("A","B"):
    norms=[row[f"{arm}_grad_norm"] for row in g]
    kl=[row[f"{arm}_ref_KL"] for row in g]
    tokens=[row[f"{arm}_action_tokens"] for row in g]
    assert len(norms)==16 and all(x is not None and x>0 for x in norms)
    metric[arm]={
        "updates":16,"mean_preclip_grad_norm":statistics.mean(norms),
        "mean_reference_kl":statistics.mean(kl),
        "total_trainable_branch_action_tokens":sum(tokens),
    }
stat={
    "CPU_ONLY_DIAGNOSIS":"COMPLETE",
    "training_metrics":metric,
    "verified_DEV_difference":a["head_to_head"]["DEV_after_A_vs_B"],
    "local_credit":a["credit_mechanism"]["credit_sign_relation"],
    "extra_local_objective_abs_coefficient_mass_fraction":
        a["credit_mechanism"]["relative_extra_local_coefficient_abs_mass"],
    "matched_after_prob":b["A_vs_B_after"]["abs_delta_selected_logprob"],
    "history_pairs":c["counts"],
    "credit_by_history":{
        "related":c["accepted_credit_records_on_related_histories"],
        "unrelated":c["accepted_credit_records_on_unrelated_histories"]},
    "pairs_use_different_onpolicy_weights":True,
    "unsupported_Qpi_samples_saved":False,
}
(OUT/"CPU_DIAGNOSIS_CONSOLIDATED.json").write_text(
    json.dumps(stat,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
ep=b["A_vs_B_after"]["different_choices_at_same_state"][0]
assert ep["role"]=="HR" and len(b["A_vs_B_after"]["different_choices_at_same_state"])==1
report=f"""# KCI-Align：16组 State-Directed GRPO / PCCE 本机 CPU 诊断结论

## 数据及检查范围

- 原始备份：{ARCHIVE.name}，SHA-256 已核对；A 组 14 项、B 组 16 项内部 manifest 校验通过。
- 读取 16 × 8 条训练轨迹/组、64 条 B 组局部信用，以及 A/B 各 96 条 DEV before/after 原生轨迹。
- 未加载 Qwen 模型、未启动 GPU、未进行参数更新。
- 两组初始 DEV 的全部 420 个决策及记录概率完全一致；原 16 次更新与独立审计已在先前 GPU 实验完成。

## 结论 1：无法证明 PCCE 优于 GRPO

| DEV 指标 | A: 状态恢复 + 终局 GRPO | B: A + PCCE |
|---|---:|---:|
| 合规任务成功 | 42/96 | 41/96 |
| CI 违规 | 19/96 | 19/96 |
| 安全未完成 | 35/96 | 36/96 |
| 平均奖励 | 0.212083 | 0.202396 |

- 相同 DEV 情景和采样种子下，A/B 有 95/96 条完整行动路径相同。
- 唯一不同发生于 {ep['scenario']}、sample {ep['sample']} 的 HR 阶段：A 选择 {ep['A_chosen']}，成功；B 选择 {ep['B_chosen']}，安全未完成。
- 两个已训练策略对“{ep['A_chosen']}”的记录概率分别约为 {ep['A_prob_of_A_choice']*100:.3f}% 和 {ep['B_prob_of_A_choice']*100:.3f}%（相差 {abs(ep['A_minus_B_prob_of_A_choice'])*100:.3f} 个百分点），同种子采样却碰巧选中了不同的动作。
- 1 条样本不能证明方法总体退化，更不能证明总体改进。

## 结论 2：已有效计算局部信用，但附加目标相对有限

- 64 条信用，覆盖 8/16 重启状态；Planner 24、Broker 24、HR 16。
- 与终局标准化优势同号 51/64、反号 13/64、最终优势符号反转 0/64。
- 有效局部项为 0.5 × centered-Qpi；在全部可训练分支动作上，按实际 1/(G × 分支数) 加权后，附加项绝对系数总量约为终局项的 {100*a['credit_mechanism']['relative_extra_local_coefficient_abs_mass']:.2f}%。
- 上述为损失系数代理，并非 LoRA 梯度范数之比；CPU 无法从归档重建真实梯度向量。
- A/B 每组均发生 16 次更新。A 平均更新前梯度范数 {metric['A']['mean_preclip_grad_norm']:.4f}，B 为 {metric['B']['mean_preclip_grad_norm']:.4f}，均为记录值而非本次重新反传。

## 结论 3：与研究目标直接相关的历史配对监督不平衡

- 8 组历史成对设置中，4 组预先指定为 Q* 动作偏好应翻转（Flip），另 4 组为保持不变（Hold）。
- **4/4 Flip 对**：仅 related 一侧通过收益门槛，unrelated 侧均没有有效 PCCE 信用；没有一组 Flip 对两侧都获得附加信用。
- Hold 对：2/4 两侧有附加信用，2/4 两侧都没有。
- 64 条有效信用里，related 历史 48 条，unrelated 历史 16 条。
- PCCE 当前训练目标是**单状态**首次行动加性信用，不是显式跨历史的 Flip/Hold 联合学习。原始终局 GRPO 则仍在 16 个状态全部训练。
- 训练依次更新每组，因此两个历史配对状态的 Qpi 评估还发生于**不同的 policy 版本**。不能把这次训练期的跨历史 Qpi 排序当成冻结同一策略的反事实效果。
- 对未通过门槛的 8 个状态，日志只留下过滤原因，没有存档全部原始 Qpi 分支收益；无法再用本机 CPU 恢复它们的完整价值差。

## 结论 4：实际策略差异极少，未观察到稳定收益

- 训练后 A/B 共 422 个相同状态、相同行动的已记录选择概率可比较；绝对 logprob 差异中位数为 {b['A_vs_B_after']['abs_delta_selected_logprob']['median']:.8f}，均值 {b['A_vs_B_after']['abs_delta_selected_logprob']['mean']:.6f}，90 分位 {b['A_vs_B_after']['abs_delta_selected_logprob']['p90']:.6f}。
- 比较的是已选动作记录概率，不能从它直接推出所有动作上的总变差距离或真实参数梯度。训练前 420 个决策的概率完全一致。
- 从部署行为角度，额外局部信用尚未形成稳定可见的变化；不能因为增加了有效信用就推断模型学会了历史条件关系。

## 按证据强弱排序的下一步

1. **先补齐观测，而非继续放大训练。** 记录每个被拒绝状态的原始 Qpi 双分支样本及拒绝原因，并对 Flip/Hold 的两侧在同一个冻结 policy 版本做并列评价。
2. **先测会不会改变策略。** 用共同的固定历史配对验证集测量更新前/后动作分布和正确 Flip/Hold，避免只报告成功率或信用条数。训练集选定的状态不能充当独立泛化结论。
3. **明确消融对照。** 将普通状态定向终局 GRPO 与额外局部信用分开；必要时引入跨历史显式训练信号，但应先验证旧方法缺失的那一环，不能从本次单种子数据直接宣告新损失有效。
4. 直到这些诊断确认额外信号能对行动产生稳定改进，再考虑追加 GPU 多种子和独立任务族评估；当前不建议把已有 B 组直接扩到 64 组。

## 可复现文件

- CPU_CREDIT_DIAGNOSTIC.json / .md：局部优势、终局符号、损失系数和 A/B 轨迹对照。
- CREDIT_DECISION_ROWS.jsonl：64 条每决策明细。
- CPU_LOGPROB_DIAGNOSTIC.json / .md：匹配状态下的动作选择概率差异。
- CPU_HISTORY_PAIR_DIAGNOSTIC.json / .md：Flip/Hold 成组信用覆盖。
- CPU_DIAGNOSIS_CONSOLIDATED.json：总览指标。
- 代码：tools/diagnose_state_matched_cpu.py、tools/diagnose_state_matched_probs_cpu.py、tools/diagnose_state_conditioned_credit_cpu.py、tools/summarize_state_cpu_diagnostic.py。

**定位：这是完整的离线机制诊断，不是新的显著性验证，也不能替代真实 4B 更新后的梯度和独立任务评测。**
"""
(OUT/"CPU_DIAGNOSIS_FINAL_CN.md").write_text(report,encoding="utf-8")
print(json.dumps(stat,ensure_ascii=False,indent=2),flush=True)
