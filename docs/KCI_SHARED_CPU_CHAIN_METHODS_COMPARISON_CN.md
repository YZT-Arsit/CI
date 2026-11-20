# KCI-Align A→R→B→C: CPU 共享可训练策略的第一轮方法比较

**研究审批范围：真实 CPU PyTorch 策略参数更新与真实 native 全轨迹回放；没有训练 Qwen3-4B/任何 LLM，不是正式 GRPO / 完整 CCE-DPR 论文算法。**

## 1. 完全固定的对照协议

受控 synthetic seven knowledge DAG graphs × three business types, with actual source-tool-mediated upstream R delivery, downstream knowledge state mutation, early privacy preflight, downstream raw disclosure/local verify, real task DB commit.

按整类图划分: 4 类图/16 个训练世界，1 图/4 开发世界，2 图/8 探索测试世界。每个世界包含合法的上游 A 证据选择、下游 B 早期保护/跳过与 source Agent 原文/私下验证、C 的真实业务提交。测试知识图在机制设计中已被检查，**不是 blind test**。

各方法共享同一个带显式可微 Horn 规则传播的角色条件化 CPU 模型，以及共同的初始化种子 (7, 1234, 2025), Adam LR 0.015、30 次共同 SFT warmstart，之后均做 120 次训练更新；输出动作名而不是列表索引。已有的源读取、合法业务路线和签名验证是各方法相同的固定工具适配器；其余关键动作由同一组神经参数决定。

### 方法含义（不能偷换成原论文实现）

- **Action SFT:** 训练世界仅一条实际可成功的 native expert trajectory，普通行动模仿训练。
- **Action SFT + CF coverage:** 加入 CCE 分叉中相同的合法观察状态和正确后续行动标签，统一 BCE/CE，不使用长期价值幅度。这是排除「只是更多数据」的必要对照。
- **Terminal REINFORCE:** 每次从共享当前策略真实采样完整 native 轨迹，仅使用本次采样动作得到的实际终局奖励；不使用另一个未选动作的 Qstar；不是 full GRPO。
- **CCE proxy:** 从同状态上游 A / 下游 B 两个合法行动分别做原生穷举，使用 exact native Qstar 之差在共享冻结 warmstart 参考的 DPO surrogate 中训练；参考Qstar不是当前模型Qpi；合并相同的 expert 下游行为示范。
- **CCE + DPR proxy:** 仅当实际采样失败轨迹中出现同状态同角色可行的下游修正时增加反馈，严格记录生成数量，不能随意对合法上游 A 追加泄漏惩罚。若 DPR 监督未产生，此组与 CCE 相等不构成 DPR 证据。

## 2. 已完成的真实执行结果

| 训练方式 | 未见图安全完成（每种子8条，均值） | 未见图CI泄漏（每种子8条，均值） | 未见图均值实际终局回报 | 开发图安全完成（每种子4条，均值） |
|---|---:|---:|---:|---:|
| Action SFT (专家一条路径) | 0.00/8 | 8.00/8 | -1.000 | 4.00/4 |
| Action SFT（同反事实状态覆盖） | 0.33/8 | 7.67/8 | -0.928 | 4.00/4 |
| Terminal REINFORCE（真实终局） | 1.33/8 | 6.67/8 | -0.720 | 4.00/4 |
| CCE 参考价值成对训练（DPO surrogate） | 0.00/8 | 8.00/8 | -1.000 | 4.00/4 |
| CCE + DPR（实际失败轨迹纠正代理） | 0.00/8 | 8.00/8 | -1.000 | 3.00/4 |
| CCE + DPR（真实失败回溯增强） | 0.00/8 | 8.00/8 | -1.000 | 2.33/4 |

同样8个未见知识图世界里，穷举 native 参考规划器实际完成 **8/8**，没有CI违规；但该参考系统还检查了每个上游证据与下游保护的所有分叉，在探索图共访问 3288 个 native 搜索状态，不能称为与神经策略等预算的公平部署对照。

## 3. 数据与参考计算成本

- 训练图 16 个源世界；expert 关键行动状态 96，同合法反事实状态上的行为标签 160，CCE A/B 同前缀行动价值比较 46。
- 取得参考标签需要真实穷举 64 个 B 下一步候选动作分支，累计 3956 个原生续行搜索状态；Action SFT 生成正确参考动作也不是免费标签。

- Action SFT (专家一条路径)：3 seeds 合计额外 terminal native episodes=0；DPR策略失败检查rollouts=0，可验证不同DPR修正对=0。
- Action SFT（同反事实状态覆盖）：3 seeds 合计额外 terminal native episodes=0；DPR策略失败检查rollouts=0，可验证不同DPR修正对=0。
- Terminal REINFORCE（真实终局）：3 seeds 合计额外 terminal native episodes=5760；DPR策略失败检查rollouts=0，可验证不同DPR修正对=0。
- CCE 参考价值成对训练（DPO surrogate）：3 seeds 合计额外 terminal native episodes=0；DPR策略失败检查rollouts=0，可验证不同DPR修正对=0。
- CCE + DPR（实际失败轨迹纠正代理）：3 seeds 合计额外 terminal native episodes=0；DPR策略失败检查rollouts=288，可验证不同DPR修正对=2。
- 回溯增强DPR：训练实际失败轨迹采样=288; 额外native修正参考查询=342; 三个seed最终保留DPR合法同状态纠正对=[4, 4, 6]; 有效角色=[{'Coordinator': 3, 'DirectoryAgent': 1}, {'Coordinator': 3, 'DirectoryAgent': 1}, {'Coordinator': 3, 'DirectoryAgent': 3}]; 未见知识图均未成功。

## 4. 科研结论应该如何解释

所有指标来自真实 full native agent actions 而非单点静态标签，独立脚本逐步重放了每条测试与开发行动轨迹，包括业务数据库变更与R的已送达记录。

本测试的样本独立性很弱：探索图只有两类规则、8个派生场景，3 seeds 不构成可信跨任务置信区间。

共享 actor 的大部分可推理性依赖预先编写的 Horn rule propagation 前向层与受控字符串解析，不是通用 LLM 在自然语言中自主发现CI。

本轮 CCE native-reference DPO 代理在未见图0/8，标准下游来源纠正DPR和回溯增强DPR也均0/8；即使回溯版本产生了4/4/6个真实失败同状态修正对，仍未让共享CPU策略跨规则家族泛化。因此目前没有CCE或DPR增量收益的证据，不能宣称方法已成立。

代码与审计：tools/chain_shared_actor_cpu.py、tools/compare_kci_chain_methods_cpu.py、tools/audit_chain_methods_cpu.py、tools/baseline_chain_native_planner_cpu.py。详细训练结果（每seed、开发/测试每轨迹）在 RESULTS.json；独立重放在 INDEPENDENT_NATIVE_REPLAY_AUDIT.json；原生参考规划器轨迹在 CHAIN_NATIVE_ORACLE_BASELINES_01。
