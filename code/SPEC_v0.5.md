# KCI-Align — Algorithm & Reward–Data Contract v0.5

状态：**可执行接口与合成环境验证通过；真实 LLM 训练、自然语言 CI 标签校准和跨任务评测尚未进行**。

## 0. 固定研究问题与边界

训练共享参数的多 Agent 执行模型，使其在消息持续交付、接收者知识状态变化的条件下，同时学会：

1. 当前接收者状态下合规且满足任务要求的披露；
2. 当前合法披露对未来能够合规完成任务的路径、成功率与实际执行成本的影响。

可信共享历史是**每个发送 Agent 获准读取的接收者交付日志**，并不向所有 Agent 开放全部私人源记录。CI 规范在单次任务中固定。多步执行中的行动可改变后续知识和任务状态。多个执行角色可以使用相同底座和相同 LoRA 权重。

选用的业务应用案例（资源分配、预约、邮件、组织协调等）可以变化；算法接口不绑定具体业务家族。H0/H1/H2 可用于受控状态变体，不作为固定的奖励等级。

## 1. 评价单位、数据单位与生成单位

**生成单位：** 某个 Agent 在特定合法观察下产生的下一步 `Action`。重要字段：`actor`, `kind`, `recipient`, `fact_ids` / semantic content, `body`, `answer`。

**状态单位：** 完整环境快照 `State`（用于运行、评分）以及权限过滤后 `Observation`（用于模型生成）。它们不能互换。`Observation` 包含角色、当前任务、用途、合法可见规范、发送者本地事实、获准读取的已交付记录和剩余时间。

**评价单位：** 从一个确切的 `(state, action)` 出发，实际执行动作，检查 `StepAudit`，再在新状态执行合规续行、检查终局任务结果。

**学习单位：** 同一 `state_id`、同一 `actor`、同一模型可见 `Observation` 下的一对不同候选行动。可以形成即时 CI、未来价值、下游修正三类偏好；每对偏好包含验证证据。

### 必需接口

```python
CandidateGenerator.generate(observation: Observation, limit: int) -> CandidateBatch
CIEnvironment.observation(state: State, actor: str) -> Observation
CIEnvironment.transition(state: State, action: Action) -> tuple[State, StepAudit]
ReferenceSearch.search(start_after_action: State) -> RefResult
ContinuationEvaluator.evaluate(state: State, action: Action) -> CCEReport
certify_preference(report_a, report_b) -> Preference | None
DownstreamPolicyRepair.repair(failed_trace: Trace) -> RepairResult
PairedReferenceComparator.compare(state, action_a, action_b) -> PairedSampleReport
export_verified_preferences(preferences) -> list[dict]
```

这些是 `kci/` 下可导入的接口。`ToyCIEnv` 提供真实可执行的有限示例；实际环境通过 `kci/interfaces.py` 的协议替换。

## 2. 候选行动生成合同

### 2.1 输入

只接受 `Observation`。生成器不可接收全局世界真值、任务 oracle、未经授权的私有事实、已经计算出的 CI 标签、参考续行结果。模型应看到其有权知道的 CI 原则和请求目的。

### 2.2 输出

`CandidateBatch(actions, exhaustive, generator_version, rejected_parse_count)`。候选行动先进行严格 JSON 结构解析，再进入真实性、权限与 CI 审计。模型声称某条消息只包含某几个 `fact_ids` 不能作为语义事实依据；`body` 必须经过独立的内容校验。测试环境采用精确的规范文本渲染，任何宣称与内容不匹配均返回 `UNKNOWN`。

### 2.3 建议生成策略

对自然语言执行 Agent，优先用单一冻结策略 
 生成多样化的下一步行动。对相同观察，尽量覆盖：

| 动作家族 | 目的 | 注意事项 |
|---|---|---|
| 完整发送 | 让下游能直接继续 | 需要验证直接披露与组合后的知识增量 |
| 选择性披露/概括 | 保留合法且有用的部分 | 须验证概括后的隐含事实与任务充分性 |
| 本地计算/受控输出 | 由有权限 Agent 处理敏感源数据 | 计算结果可能仍能反推原始秘密 |
| 请求条件或授权 | 获取下一步合法执行所需信息 | 须验证请求确实改变任务状态、并有完成路径 |
| 合法委托/替代接收方 | 改变责任角色或发送路径 | 验证新接收者的规范与工具权限 |
| 暂缓/停止 | 在当前无法继续时使用 | 不能因为不泄露就自动获得任务成功奖励 |

这些是候选动作的**覆盖维度**，不是强制每个状态必须产出六种动作。建议生成时用 token-level 多样采样和任务条件化动作 proposal；按标准化执行后果去重。对于没有生成合规替代的状态，可追加采样或标为数据覆盖缺口，不把未找到当作不存在。

测试版 `ToyCandidateGenerator` 穷举它定义的有限动作集合，因此允许作有界精确比较；`JSONCandidateGenerator` 永远报告 `exhaustive=False`。

## 3. 即时 CI 验证标准（在 CCE 中执行）

### C1 — 行动可执行性

当前 actor 与调度角色一致；工具、数据来源、消息接收者符合该 Agent 当时权限；不得生成模型实际上拿不到的事实。如果不满足，返回 `INVALID_ACTION`。无效消息不能改变环境状态。

### C2 — 真实消息内容审核

结构化内容经可信渲染与实体/关系验证；自由文本由独立 SemanticVerifier 提取并验证实际表达的事实。未验证的 paraphrase/隐含属性返回 `UNKNOWN`。模型提供的元数据、步骤解释和隐私自评分均为不可信输入。

### C3 — CI 传输规范

按发送者、信息主体/属性、接收者、任务目的和传输原则评估直接披露。接收者此前已知道的受保护事实仍可能构成本次违规传输。

### C4 — 接收者知识转移

先记录实际**成功交付**事件，之后更新该接收者的知识闭包：

\[
K_{t+1}(r)=\mathrm{Closure}\left(K_t(r)\cup\mathrm{Semantics}(m_t)\right).
\]

未送达的草稿、模型内部计划、外部不可信评论都不进入交付日志。训练时可以模拟违规动作并记录后果；真实系统中的外部发送须与训练沙盒隔离。

### C5 — 新增未授权推断

检查 \(\Delta K=K_{t+1}(r)\setminus K_t(r)\) 是否产生特定上下文下未获授权的关系。推断证据附带历史消息、当前消息及明确的事实组合规则。仅凭猜测无法确定的开放世界推断先返回 `UNKNOWN`；不能当作确定的违规标签。

即时审计输出 `VALID`、`VIOLATION`、`INVALID_ACTION` 或 `UNKNOWN`，并附直接违规及组合推断依据。若 `VIOLATION`，不进行“合规参考续行”搜索；此行动可作为负候选，但要配合经验证能完成任务的正候选。

## 4. 参考续行的验证标准

### R0 — 可重放的同前缀比较

两候选必须起始于完全相同的世界状态和当前 Agent 观察。保持任务规则、接收者历史、外部事件分布、剩余交互期限和参考模型版本一致；只替换当前动作。后续 Agent 在不同的状态下自然产生不同决策。

### R1 — 每条参考动作有执行与可见性凭据

每一步重新执行 C1–C5。参考计划可以由强模型或规划器生成，但每个子行动必须能由其执行角色**合法可见的信息和工具能力**支持。不能把全局 oracle 注入模型输入。

### R2 — 终局任务结果可验证

参照 `utility.json` 式任务 oracle 或具体环境执行器，判断任务结果、必需信息覆盖、责任角色和接收者、时间期限。自然语言结果依赖 Judge 时，必须给出证据级别；`UNKNOWN` 不允许被记录成成功。

### R3 — CI 全轨迹审计

所有实际交付的消息均按到达顺序验证。只要有一次违规，轨迹即不属于合规完成路径；后续文字上的修正不能撤销已发生的信息流。

### R4 — 成本真实计量

仅在合规成功轨迹中比较执行成本。成本字段包括消息、工具调用、等待或实际资源消耗。演示环境使用单位动作成本；真实任务不得无依据地用回复长度代表真实任务价值。

### R5 — 参考搜索证据等级

| 标记 | 需要什么证据 | 可以做什么 |
|---|---|---|
| `exact_within_horizon` | 受限有限动作空间、转移/验证完备、搜索预算足以覆盖全部声明动作 | 得到该有界模型中的精确最小完成成本；可比较未来行动 |
| `proven_unreachable_within_horizon` | 同上，且穷尽后确认期限内没有合规成功路径 | 可与另一行动的成功见证比较可达性 |
| `verified_witness_only` | 找到一条完整的合规成功轨迹，但未穷举 | 证明存在一条路径；不能单凭该路径成本排除别的高价值路径 |
| `unknown` | 搜索预算耗尽、语义无法验证、模型未找到路径等 | 只记录缺口；不得产生硬负偏好 |
| `paired statistical estimate` | 使用冻结参考策略、同前缀与配对随机种子，估计任务成功率/成本差并有误差控制 | 可产生带统计证据的参考策略价值偏好；不声称为最优值 |

### R6 — 统计比较（面向无法穷举的环境）

冻结参考执行策略 \(\pi_C\)，对同一状态的两合法行动 \(a,b\) 使用相同的外部随机种子序列；每个分支重新运行下游策略。定义 \(Y^a_j=1\) 当且仅当完整续行同时满足 CI 和任务成功。构造配对差 \(D_j=Y^a_j-Y^b_j\in[-1,1]\)。

\[
\bar D=\frac{1}{n}\sum_j D_j,\qquad
\epsilon_n=\sqrt{\frac{2\ln(4/\delta)}{n}}.
\]

在**独立采样种子、正确审计、固定参考策略**的假设下，利用 Hoeffding 界（成功率与成本至多两项比较，以 Bonferroni 分配整体错误概率）：若 \(\bar D>\epsilon_n\)，可以产生在该固定参考策略下 \(a\) 的安全成功率更高的统计偏好；若 \(\bar D<-\epsilon_n\) 则相反。否则不给出严格排序。

仅当所有配对样本中两条分支都合规成功，才进一步对归一化的配对成本差采用相同的保守界。实际使用可增加更精确的置信区间与顺序停止规则，但必须预先固定以避免反复检验选择性偏差。

接口：`PairedReferenceComparator`。它生成 `epistemic_status = statistical_estimate_NOT_optimality_proof`。统计比较针对 \(Q^{\pi_C}\)，不是全局 \(Q^\star_{\mathrm{safe}}\)。当参考策略在一个状态很差时，该差异也可能体现下游执行能力；需结合 DPR 与已验证的合规见证解释。

## 5. 奖励/偏好评价函数与数据匹配

每条轨迹保留三元向量：

\[
J(\tau)=(\mathrm{CICompliant}(\tau),\ \mathrm{TaskSuccess}(\tau),\ -\mathrm{Cost}(\tau))
\]

比较次序：**CI 合规 → 任务成功 → 合规成功时的成本**。这消除了 H0/H1/H2 固定奖励系数的必要性。若使用 RL 标量，需要预先规定 CI 约束或在有界评分范围内一致映射；当前 DPO 导出的是证据支持的偏好而非手工加权回报。

| 训练信号 | 必需字段 | 现有数据的帮助 | 缺失则怎么处理 |
|---|---|---|---|
| 即时 CI | 已交付历史 + 当前文本语义 + 当前规范作用域 + 接收者、目的、来源 | PiSAs 可提供静态属性规范与角色；MultiAgentPrivacy 可提供关系组合结构 | 不生成 CI 正负标签 |
| 任务充分性 | 任务 oracle + 执行后的实际目标状态 + 可验证完成条件 | PiSAs `utility.json` 已有终局决策规则 | 只记录 CI 审计，不生成“更优完成路径”偏好 |
| 未来参考价值 | 完整状态转移/工具规则 + 后续可执行候选 + 固定期限 + 参考验证器 | 需要新增 KCI 执行模拟或可重放任务 | `unknown`，不把搜索失败当负例 |
| 下游修正 | 真实失败轨迹 + 同决策状态 + 同权限替代行动 + 已验证合法续行 | PiSAs 的运行程序可生成 A2A trace；尚需动作级分叉及重放 | 不生成 `repair` 样本 |
| 无关历史鲁棒性 | 同一世界、等价事实/规范条件下的可比观察 | 需从原始任务生成受控历史变体 | 可用于测试，不强制成为新的 reward 类别 |

**不能直接继承的标签：** PiSAs 的 `appropriateness` 是给预先定义的属性的标签；`timeline` 是预置资料；MultiAgentPrivacy 的 `sensitive` 是表级标签。它们均没有提供我们新构造的 `(delivered_history, current_action, derived_fact, recipient, purpose)` 组合级 CI 训练标签。

**数据链路验收：** 每条偏好导出前，检查 `group_id/world_id`, `state_id`, `actor`, `authorized_observation`, `action_a/b`, `CI_verdict`, `task_verifier`, `evidence_status`, `policy_version`, `reference_policy_version`, `search_budget`, `source_norm_version` 和训练/测试划分。按基础世界、任务规则家族拆分，防止相关样本跨集泄漏。当前 v0.5 代码导出的 metadata 已实现核心状态与证据记录；面向真实论文数据的每个来源版本和分组字段仍需在外部数据管道补齐。

## 6. CCE 算子流程

```
输入 state, actor, action, current_policy_snapshot, candidate_generator, reference_engine
  observation <- Observe(state, actor)
  next_state, instant_ci <- sandbox_transition_and_audit(state, action)
  if invalid or unknown: output unresolved
  if direct or incremental CI violation: output violating_action (no safe reference suffix)
  actual_rollouts <- current shared policy at next_state (optional, record policy version)
  reference <- bounded safe search at next_state (or paired rollout from frozen pi_C)
  output CI evidence, actual outcome, ref evidence, uncertainty, verifier and model versions
```

`certify_preference` 优先处理“经证实能合规成功的行动” vs “当前确定 CI 违规的行动”；两个都当前合规时，在精确有界参考搜索或可证明的可达性差异下排序。`PairedReferenceComparator` 为非穷举情况提供单独的统计接口。未经认证的二者返回 `None`。

## 7. DPR 算子流程

```
输入 实际失败轨迹 (每个状态、角色、行动、交付及审计)
  找出首次已经发生的 CI 违规；后续步骤不能修复该已发生的泄露
  从该位置向前回溯(若无违规，回溯任务失败末端)
  对每个可修正状态：
      只让当时的 actor 依据当时的 authorized observation 提议替代行动
      对原行动和替代行动调用同一 CCE
      要求替代行动的可执行性、CI 与任务充分性经过验证
      需要严格差异：直接 CI 错误，或有足够证据的后续合规价值优势
      输出 (same_prompt, chosen_repair, rejected_original, supporting_evidence)
  搜索无可靠结果时 unresolved；不为最早合法行动编造违规责任
```

DPR 的输出 `RepairResult` 会携带修正步骤索引、当前 Agent、行动偏好与证据。共享参数训练时，所有角色偏好可写进同一模型的训练集。任务成功但低效的轨迹由 CCE 的长期比较负责，DPR 专注真实失败状态。

## 8. 生产环境扩展的关键规则

1. **候选质量控制：** action-category coverage 只用作提议策略，最终评价依赖完整动作后果；生成器严禁读取评判 oracle。
2. **消息语义验证：** 真实自由文本需要独立验证；声明的 fact IDs 与原文不一致则 `UNKNOWN`。复杂概率推断必须允许标签不确定。
3. **状态重放：** 当前前缀含所有真实交付、消息接收者、工具副作用与可见性版本。没有重放能力就不能做公平的 same-prefix 分叉。
4. **参考规划器权限：** 规划器可以用于离线搜索，但候选 Agent 行动必须符合执行者可见信息和工具权限；后续不允许偷看隐私 oracle。
5. **不重复惩罚：** 当前触发违规的行为与更早合法行为分开评价；早期行动可根据经过验证的未来路径成本差接受长期偏好。
6. **明确不可行与搜索失败：** 有界证明与搜索未知有不同的训练含义；未搜索到任务成功不能自动产生负样本。
7. **固定版本：** CCE 比较时记录模型快照、参考续行版本、CI 标注版本、任务 oracle、采样种子与预算；避免新旧策略混跑造成伪差异。
8. **效用一致性：** 本地计算、委托和请求信息的行动都需由终局任务验证器评价，不能因为“看起来安全”自动赋予任务进展奖励。
9. **防止保守捷径：** 合规正样本需要真实任务完成证据；合法的敏感关联及相似但无关历史作为鲁棒性对照；空回答通常无法获得成功证据。
10. **训练划分和许可证：** PiSAs 原始数据的静态角色/规范在遵守许可证条件下复用；新增动态组合标签有独立来源与审核版本。不要把 MultiAgentPrivacy 的目标标签冒充 CI 授权。

## 9. 已执行的接口验证

`PYTHONPATH=. python -m unittest discover -s tests -v`

涵盖：

- 同一候选消息在不同交付历史下由 `VALID` 转为 `VIOLATION`；
- 相似但不相关的标识符无法组成违规关联；
- 信息真正交付之前不会修改原状态；
- 直接违规与增量违规分别被识别；
- 角色权限和接收者授权日志分离；
- 模型声明内容与真实消息不匹配时返回 `UNKNOWN`；
- 过度拒绝不能击败已验证的合规任务成功；
- 不完整搜索返回 `UNKNOWN`，仅找到续行时返回 `WITNESS_ONLY`；
- 有界精确续行产生长期偏好；
- 配对参考策略的统计成本比较标注估计属性；
- DPR 在 B 的违规决策处产生修正，而非回溯责备 A 的合法行为；
- PiSAs 静态标注被阻止直接进入历史训练。

这些是**逻辑与接口单元测试**，尚无模型真实训练和数据统计结论。

## 10. 后续要填充的模块（研究主线不变）

1. 用真实自然语言 Agent 环境实现 `CIEnvironment.transition`, `observation` 和 `TaskOutcomeVerifier`；消息可重放且副作用隔离。
2. 构造经过校验的动态历史与关联真值；先验证标签质量、行动可执行性及合规任务是否存在。
3. 将 Qwen 模型接入 `JSONCandidateGenerator`，用独立语义验证与结构化 CI 规范审计实际生成的 body。
4. 确定参考搜索预算及固定参考策略，生成带置信证据的长程偏好，与初始逐步 CI 偏好做严格消融。
5. 导入 TRL DPO/后续 RL，更新共享参数执行模型，再在真实多步、多业务测试上观察合规、完成率、效率与鲁棒性。
