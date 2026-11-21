# KCI-Align：2026-11-21 CPU 直接研发结果

**性质**：已训练、完整原生回放验证的 CPU 机制实验；不是 Qwen3-4B 实验，不是独立盲测，也不能直接宣称 ICML 方法优势。

## 一、问题与机制：保持原 KCI 研究主线

在已有的动态 A→R→B→C 原生工作流中，RecordsAgent A 实际发送当前合法的来源证据，接收者 R 真实积累知识，Coordinator B 与来源 Agent 后续行动并完成实际模拟业务。我们不修改该环境的原始转移和规范。

原共享小型 Neural-Horn actor 的问题：训练图上 A 14/14 正确，开发者已检查过的探索测试图上 A 严格最优仅 2/6，且部分错误 logits 高度自信。直接在共享网络上叠加长期偏好梯度，可能破坏已经学会的下游合规行为。

本轮实现了独立的精确策略条件化续行评价：固定同一个当前 Agent 策略，强制当前一个合法行动，递归枚举后续真实 native 行动，根据每个后续状态的模型 softmax 概率计算 Qpi。它不是 ideal Qstar 搜索，也不是静态标签替代真实执行。

原冻结 SFT-CF 小策略每个 seed：
- Train 14 条严格 A 选择中的 Qstar/Qpi 排序冲突为 6；
- Dev 4 条严格 A 选择中为 2；
- Developer-seen test 6 条严格 A 选择中为 2。

这证明在本环境中，理想最优续行与当前团队能执行的实际续行并不总一致。

## 二、直接共享参数的 Qpi 学习为什么失败

同一 16 train /4 dev/8 developer-seen exploratory worlds；3 seeds；共同 30 次专家 SFT warmup 后各 120 次更新。所有方法使用相同原生任务和公开资源合同。

| 训练方法 | 三种子探索测试贪心安全完成 | 上游严格最优 | 平均随机策略期望回报 |
|---|---|---|---:|
| 共享 Action SFT | 8/8,8/8,8/8 | 2/6 | 0.41299 |
| 共享反事实覆盖 SFT | 8/8,8/8,8/8 | 2/6 | 0.53613 |
| 共享参考 Qstar-DPO 代理 | 8/8,8/8,8/8 | 2/6 | 0.34330 |
| 共享联合 Qpi-DPO | 4/8,4/8,8/8 | 2/6 | 0.00146 |
| 共享下游先行 Qpi-DPO | 8/8,4/8,4/8 | 2/6 | 0.06443 |

失败轨迹表明，一些原本正确的 OPEN_PRIVATE_CHANNEL → VERIFY_PRIVATE_R 被改成 SKIP_PRIVATE_CHANNEL → SEND_RAW_R，发生真实原生 CI 违规。立刻把合法反事实偏好混进共享参数不是可靠修复。

另外测试了“仅在原共享 logits 上添加 A 残差”四种方法。安全性得到保留，但错误 A logits 十分自信，4 种方法均没有突破 A 的 2/6。此负对照完整保留在 artifacts/20261121_kci_role_local_credit_cpu_v1。

## 三、替换过度自信的上游 A 头，严格冻结下游

对原共享 SFT-CF 策略冻结 B、来源 Agent、C；仅对 RecordsAgent A 的两个合法来源包训练一个独立动作评分头，输入只包含合法公开规则推理、包成本和公开 setup 合同，不包含隐藏 Qstar/Qpi 或将来的正确行动。

| A 层训练方式 | A 严格最优，三个 seed | 上游平均 Regret | 策略期望回报均值 | 策略 CI 违规概率均值 |
|---|---|---:|---:|---:|
| 最优动作 SFT | 4/6,4/6,4/6 | 0.007 | 0.62415 | 1.589% |
| Qstar 偏好 DPO | 6/6,6/6,6/6 | 0 | 0.60802 | 2.358% |
| Qpi 偏好 DPO | 4/6,4/6,4/6 | 0.007 | 0.62900 | 0.896% |
| 两个价值教师同向 DPO | 4/6,4/6,4/6 | 0.007 | 0.62737 | 1.058% |

所有方法三种子均贪心 8/8 安全完成。重要：使用 ideal Qstar 训练可以降低上游相对于 ideal 下游最优的 Regret，但对尚未充分胜任的当前下游策略，Qpi 对应的实际策略期望回报反而较高。这不是声称 Qpi 总是比 Qstar 好，而是测到了策略能力条件化的作用。

## 四、下游能力修正 → 重新计算 Qpi → 再训练上游 A

首先尝试直接替换 B/来源 Agent 头：尽管贪心安全，随机策略期望回报降至约 0.374，所以该方案被明确保留为负结果。

改成**保留原共享底座所有动作分数，仅针对下游 B/来源 Agent 叠加小型可训练残差**；利用原有 train 反事实专家状态做 120 次局部修正。冻结后，重新估计 A 同状态两个合法选项的准确 Qpi；最终对 A 独立动作头做 80 次 DPO。A 的优化不会改变 B/来源/C 的任何 logits；B/来源修正也不会改变 A 的 logits。

| 阶段/方法 | A test 严格最优（三种子） | 贪心安全完成（三种子） | 随机策略期望回报均值 | 随机 CI 违规概率均值 |
|---|---|---|---:|---:|
| 原共享 SFT-CF | 2/6,2/6,2/6 | 各8/8 | 0.53613 | 5.370% |
| 下游残差修正，不动 A | 2/6,2/6,2/6 | 各8/8 | 约0.62810 | 约0.188% |
| 下游残差修正 + A-Qstar DPO | 6/6,6/6,6/6 | 各8/8 | 约0.64535 | 约0.083% |
| **下游残差修正 + A-Qpi DPO** | **6/6,6/6,6/6** | **各8/8** | **0.64538** | **约0.083%** |

最后一行各 seed 完整策略精确期望回报：0.647315 / 0.645201 / 0.643628。A 参考 Regret 全部为 0。期望回报是小型 CPU actor 随机行动概率与原生状态转移的精确枚举，不是单条贪心轨迹。

下游修正造成 A 同状态 Qstar/Qpi 排序冲突的变化：
- Train 14 条严格状态：每个 seed 6 → 0；
- Dev 4 条严格状态：每个 seed 2 → 0/2/0；
- Developer-seen test 6 条严格状态：每个 seed 2 → 0。

这支持一个可检验的机制：**长期信用必须结合下游实际能力；先恢复下游合规执行能力，再训练上游，能够在受控 CI 环境中兼得更高策略期望回报与更好的上游长期最优行动。**

最终上游 Qpi-DPO 与 Qstar-DPO 结果几乎相同，不能宣称 Qpi teacher 在下游修复后仍独立优于 Qstar。此次真正得到的结果是能力变化、价值排序与训练信用的阶段性耦合。

## 五、验证、源码和实验位置

主要源码：
- tools/kci_exact_policy_continuation_cpu.py：精确冻结策略 Qpi 的完整原生续行评价；分别区分立即合法 CCE 与实际违规 DPR。
- tools/run_kci_cpu_policy_conditioned_gate.py：五种共享 actor 方案 × 三 seed。
- tools/run_kci_role_local_credit_cpu.py：上游残差负对照。
- tools/run_kci_calibrated_upstream_cpu.py：A 替换头的 Qstar/Qpi 对照。
- tools/run_kci_downstream_first_cpu.py：下游直接替换负对照。
- tools/run_kci_downstream_residual_cpu.py：下游残差→A 价值学习。
- tools/audit_kci_cpu_direct_research_20261121.py：从保存权重独立重载并原生回放。
- code/tests/test_kci_exact_policy_continuation_cpu.py：核心策略续行、合法性、DPR 隔离和冲突复现。

所有原始记录和权重保留在 artifacts/20261121_kci_* 目录。最重要的是：
- artifacts/20261121_kci_downstream_residual_cpu_v1/RESULTS.json
- artifacts/20261121_kci_downstream_residual_cpu_v1/INDEPENDENT_SAVED_MODEL_REPLAY_AUDIT.json
- artifacts/20261121_kci_downstream_residual_cpu_v1/EXACT_QPI_RANK_CONFLICT_AUDIT.json

测试：相关 4 组 pytest 共 **15 passed**。最终 6 个保存模型重新加载、72 条非训练完整原生轨迹、72 个 soft-policy 期望回报以及训练/开发/探索状态下的角色动作不变性，均通过独立回放。每个最终检查点有 SHA-256。

注意实验脚本拒绝覆盖已有结果目录，复现应在干净工作树执行或先单独备份原始 artifacts。

## 六、重要边界与 ICML 下一步

此结果是 CPU 合成机制原型，**不是 ICML 已经成立的方法**，理由包括：

1. 测试仅来自两个已经参与开发的 Horn 图家族（8 个高度相关派生世界）；3 seeds 不等于 24 个独立任务；
2. 已提供显式 public Horn 推理层和手工风险成本特征，不能说语言模型自己学会自然语言隐私推理；
3. 下游残差阶段使用的是已有原生专家 SFT 状态，不是初稿中基于实际失败轨迹的完整 DPR-DPO；不能直接更名为已完成 DPR；
4. 精确 Qpi 只能在小型 CPU 状态树上计算，真实 LLM 必须采样后续轨迹且核算额外 token/工具成本。

**下一步研究方法定义**：相同底座 LLM，用至少两组可路由的角色 LoRA；先在后续真实失败状态上进行局部 CI 纠错，冻结已恢复能力的下游角色，使用相同 A 状态两合法行动的冻结团队真实续行生成可信 Qpi 偏好，再优化上游角色。严格比较 SFT、状态覆盖 SFT、联合 Qstar/CCE、直接联合 Qpi、下游纠错→A-Qstar、下游纠错→A-Qpi，实测 CI 泄漏/任务成功/真实随机策略价值/额外 rollouts 成本，保持未见业务机制的独立封存。

研究主线保持 **动态接收者知识 + 长期合法行动价值 + 实际下游责任归因**，不再反复寻找 Motivation，也不以合成 CPU 数值直接替代真实 LLM 论文实验。
