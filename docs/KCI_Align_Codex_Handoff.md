# KCI-Align — Codex 开发交接合同 v0.6

**定位：** 配合 `KCI_Align_Research_Proposal.md` 与 `\code` 使用。  
**固定研究主线：** 多 Agent 执行模型在接收者知识实际积累的过程中，学习即时 CI 披露适当性与长期合规任务完成策略。CCE、DPR、信用分配是训练工具。  
**当前证据：** v0.5 toy 环境 24/24 单元测试通过；尚无真实 LLM 训练或论文指标。

## 一、可直接交给 Codex 的初始指令

> 请以随附的 KCI-Align Research Proposal v0.6、SPEC_v0.5.md 和 v0.5 Python 原型为唯一的项目方法合同，分阶段完善真实多 Agent 环境与共享参数执行模型训练。先阅读整个实现、运行 24 项测试和 demo，并给出代码审计：文件职责、当前假设、接口缺口、可复现结果。研究目标固定为“接收者知识积累下的长期 CI 策略学习”，不得擅自将研究改成一般多 Agent 隐私过滤器、独立信用分配算法、静态 CI benchmark 或仅运行时 Guard。
>
> 核心接口沿用 `Observation`, `Action`, `State`, `StepAudit`, `CandidateGenerator`, `ContinuationEvaluator` (CCE), `DownstreamPolicyRepair` (DPR), `ReferenceSearch` 与 `PairedReferenceComparator`。任何新增模块都必须写明对应的研究难点、需要的输入字段、训练信号和测试验收标准。所有文本行动必须审查实际 `body` 的语义，不能相信模型自报的 `fact_ids`。任何参考路径的评分必须区分 exact / witnessed / statistically estimated / unknown，未找到不能解释为不可达。每个阶段只修改实现与测试，不改变 Proposal 固定研究问题、合规优先的轨迹评价或证据等级。
>
 请先只完成**阶段 0：审计与接口差距报告**；输出逐文件摘要、真实依赖、运行日志、测试结果、最小改动方案和潜在奖励—数据错配。在审核通过后再开始实现阶段 1。

## 二、依赖顺序与里程碑

### Phase 0：现有代码与研究合同对账（先完成）

输入：原型 code、Proposal 和 SPEC。要求记录、运行 Python 环境、可复现测试与演示、SHA256 文件清单、现有模型/任务/评分硬编码的位置。

**验收：** 24 项基线测试无回归；明确 `toy_env` 与未来 LLM adapter 的差异；不声称已经存在真实 CI 数据标签或真实模型改进。

### Phase 1：可重放的真实多 Agent 状态转移

实现 `observation(state, actor)`、`transition(state, action)`、`snapshot/restore`、事件投递 ID、角色 ACL、工具权限、任务期限、终局 oracle 及执行成本。将 PiSAs 的预置 `timeline` 导入初始事实，**不要计入 `delivered_events`**。只有真实 `DELIVERED` 事件更新某接收者的知识状态。

**验收：** 同一前缀克隆分叉无跨分支副作用；角色各自看不到无权访问的原始源数据；同一消息在两种合法历史下可产生不同 CI 结果；执行顺序变化仍按对应历史判断。

### Phase 2：自然语言语义—CI 验证器

对自由生成 `body` 做实体/关系解析、可检查蕴含、来源追踪以及直接/衍生事实授权判断。区分事实不确定与规范不确定，输出 `VALID / VIOLATION / INVALID_ACTION / UNKNOWN`。记录历史触发证据和当前消息证据。允许合法组合，不可把“有实体关联”直接当违规。

**验收：** paraphrase、相似但不同的标识符、合法推断、重复禁止披露、隐藏 oracle 泄漏、条件不明确等反例通过。无法自动确定的语义进入人工复核队列。

### Phase 3：CCE 参考续行质量

先在有限有界任务上实现穷举比较，再实现冻结参考策略的配对种子续行；每一步用 Phase 1/2 的权限和 CI 验证器复核。保存 `policy_version`, `ref_version`, `search_budget`, `evidence_status`, `confidence`、成功见证与成本。`witness_only` 不产生“比另一动作的未知路径更优”的硬偏好。

**验收：** 有界穷举 vs 限额搜索行为清晰区分；配对统计只在通过预设置信条件时导出偏好；估计不确定时返回 None；能独立复现分支对比。

### Phase 4：DPR 可验证的下游修正

从失败轨迹识别首次违规或确认的任务失败；在相关真实状态和真实 Agent 观察下生成替代行动；调用 CCE 验证其合规与任务充分性。允许多个可能的正确修正；找不到可靠修正返回 `unresolved`。

**验收：** 上游当前合法、下游错误时在下游生成修正样本；上游另有更优合法路径时单独生成长期偏好；不能把参考策略越权看到的答案用于执行。

### Phase 5：奖励—数据合同与共享模型训练

按基础世界与规则家族分组划分数据；生成同 Prompt 的 `chosen/rejected` 及证据权重；使用冻结 DPO 基准模型 \(\pi_0\) 和冻结参考续行策略 \(\pi_C\)，二者独立管理版本。采用一个 4B–8B LLM 共享参数、角色条件化和 LoRA，先与轨迹 SFT、逐步 CI 奖励 DPO 比较。

**验收：** 随机种子、数据/模型版本、reward 统计、合法成功率、任务遗漏率和模型权重可追溯；训练前必须通过数据合同检查，不得把 PiSAs 属性标签直接当新派生关系标签。

### Phase 6：完整评测与消融

对 \(\mathcal D_{\mathrm{CI}}\)、+\(\mathcal D_{\mathrm{future}}\)、+\(\mathcal D_{\mathrm{repair}}\) 逐项消融；加入历史无关扰动、合法组合正例、交换消息顺序、不同接收者和跨业务域任务；与真实强基线公平比较。

**验收：** 报告 Safe Task Success、Direct/Incremental Violation、Task Success、合法完成成本、过度拒绝、参考证据覆盖和训练成本；提供失败案例追踪与多种子结果。

## 三、数据合同—奖励字段映射

| 奖励所需事实 | 字段或接口 | 来源/生成方式 | 丢失或不可信时 |
|---|---|---|---|
| 当前 Actor 能看到什么 | `authorized_observation`, `role_acl` | Env 运行时权限过滤 | 拒绝构造偏好 |
| 接收者真实历史 | `delivered_events` | Env 交付状态机 | 不能计算历史增量 |
| 信息语义 | `actual_message_semantics`, `provenance` | 独立语义验证器 | `UNKNOWN` |
| 受保护推断的授权 | `derived_fact_norms`, `purpose`, `norm_version` | 明确的 CI 规则/人工仲裁 | `UNKNOWN` |
| 任务正确性 | `task_oracle`, `task_state` | 任务执行器/可验证规则 | 不生成长期价值偏好 |
| 合规续行 | `trace`, `witness`, `evidence_status` | 参考搜索/冻结策略 | 标为不确定 |
| 训练偏好 | `state_id`, `actor`, `prompt`, `chosen`, `rejected` | CCE/DPR 高可信筛选 | 排除该对 |

## 四、必须保持的禁止事项

- 不能为三类历史 H0/H1/H2 分配人为固定奖励。它们只是可选控制变量。
- 不能只用词面敏感性判定 CI，也不能把所有推断都当违规；接收者/用途/传输原则必须进入判断。
- 不能因为搜索没有找到路径就给行动错误负标签。
- 不能用一个不固定或不断改变的参考模型直接比较不同分支的长期价值。
- 不能让真实消息发送或工具写入发生在评分用的分叉沙盒之外。
- 不能以原始 PiSAs `timeline` 或其 `appropriateness.json` 替代新历史条件下的规范真值。
- 不能把“同一底座模型”误写成“各 Agent 共享数据”；共享参数与共享授权日志的权限范围不同。
- 不能以玩具环境 4/7 步或 24 项单测作为真实 LLM 的收益证据。

## 五、每阶段汇报模板

1. **完成内容：** 修改了哪些代码文件与接口，新增什么配置或数据字段。
2. **为什么需要：** 对应 Proposal 的哪个学习问题、哪条奖励/数据合同。
3. **验证：** 自动测试、反例测试、可重放 demo、耗时/调用量。
4. **证据级别：** 精确、有界、见证、统计估计、不确定。
5. **问题与偏差：** 哪些数据/奖励标签尚不可靠，是否影响训练结论。
6. **下一阶段：** 最小必要工作，不擅自扩展新的研究主线。

---

**起始命令（在解压的 v0.5 根目录）：**

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python examples/run_demo.py
```

**研究决策权限：** 代码架构与测试可由 Codex 推进；新的研究问题、CI 规范真值、奖励目标改变、额外训练损失和“论文创新点”需要单独进入研究审查。
