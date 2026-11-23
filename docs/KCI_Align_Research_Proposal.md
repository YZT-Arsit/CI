# KCI-Align：动态接收者知识条件下的长期语境完整性策略学习

**Research Proposal · v0.7 · 2026-11-23**
**English working title:** *KCI-Align: Learning Long-Horizon Contextual Integrity under Evolving Recipient Knowledge in Multi-Agent Systems*  
**研究类型：** 方法论文假设（模型训练）｜**阶段：** 真实共享 4B 执行与训练已打通，CCE/DPR 独立增益未建立
**定位：** 研究计划及 Codex 交接依据；不把待检验假设写成既有实验结论

## 摘要

在多智能体协作中，信息通常分散于不同执行者，并通过多轮消息逐步抵达接收者。某项披露在当前语境下合规，仍可能改变接收者的知识状态，进而约束后续信息流并影响任务完成路径。例如，两条分别适当的业务消息在连续交付后，可能使接收者获得原本未经授权的身份—属性关联。已有 Contextual Integrity（CI）研究从信息流判断、运行时控制、隐私偏好训练和轨迹隐私等方向处理了这一问题的不同部分。我们研究的核心是：**如何直接训练执行 Agent 背后的语言模型，使其将动态接收者知识用于当前披露决策，并选择有利于未来合规完成任务的行动？**

拟议方法 KCI-Align 维护权限范围内的可信交付记录，对实际消息的直接披露和新增未授权推断进行状态条件化审计；通过 CI-Conditioned Continuation Evaluation（CCE）比较当前合法行动的后续合规任务价值；通过 Downstream Policy Repair（DPR）将下游可修正错误转化为对应执行位置的行动偏好。最终以共享参数的角色条件化模型学习即时 CI、长期协作价值与下游修正三类监督。项目将重点检验长期监督相对逐步 CI 奖励的增量、信息到达顺序与历史相关性带来的学习难点，以及任务效用—隐私之间的实质性权衡。当前已有可运行的有限合成环境和接口测试，尚未形成真实模型的性能结论。

**关键词：** Contextual Integrity；Multi-Agent LLM；Stateful Privacy；Sequential Policy Learning；Recipient Knowledge；Preference Optimization

## 1. 固定研究问题与研究假设

### 1.1 问题定义

给定一项由多个 Agent 协同执行的用户任务，Agent 持有不同来源的信息，按角色和工具权限向其他 Agent 或外部接收者发送消息。接收者随着消息的**实际交付**积累知识。当前信息流的适当性由信息主体、发送者、接收者、用途、属性及传输条件共同决定；已交付历史可能改变当前消息所产生的可推断事实，也可能改变后续合规完成任务所需的行动路径。

我们希望直接优化执行模型参数，使其能够：

1. 在接收者当前知识状态下，生成合规且任务充分的信息披露；
2. 在多个当前合法行动之间，选择能够更好支持后续**合规任务完成**的行动；
3. 当下游 Agent 未能正确利用合法路径时，获得针对该决策位置的可验证训练反馈。

**研究主线固定：** 动态接收者知识 × 多步多 Agent 协作 × CI 行动学习 × 共享执行模型参数。首版规定一次任务中的 CI 规范固定；共享的是发送者获准查询的可信交付记录，各 Agent 的私人源数据仍有权限隔离。业务场景不限于会议、Jira、医疗或项目分配，具体案例用于数据与评价落地。

### 1.2 核心研究假设（待验证）

- **H-A：** 仅根据当步是否泄露/任务是否推进进行训练，可能无法稳定学习合法早期行动对未来合规完成路径的影响。
- **H-B：** 同前缀分叉、经 CI 验证的参考续行，可在部分状态中提供比即时 CI 标签更有区分度的长期行动偏好。
- **H-C：** 将下游可修正错误定向送回对应决策状态，能减少把下游策略失败误传为上游合法行动的负面监督。
- **H-D：** 在明确的任务充分性与合法组合正例下，模型可以改善合规完成率，而不依赖一律拒绝、历史越长越保守等捷径。

这些假设的成立程度决定方法贡献。接口测试与真实模型能够更新均不等于 H-A—H-D 成立。v0.7 保留研究目标，停止同标签七状态的重复方法训练；新增监督通过验收后，才进入新的方法比较。

## 2. 研究动机与相关工作定位

### 2.1 为什么动态接收者知识重要

设接收者已合法获得“编号 X17 属于 Alice”，另一个 Agent 准备发送“X17 对应某受限业务属性”。在预设规范下，两条原子消息可能分别允许，但组合后出现未经授权的身份—属性关联。把交付顺序互换，最终的受保护关联依旧可能出现；发生违规的时间和需要改变行动的 Agent 会变化。提前发送一条当前合法消息，还可能令后续任务必须通过本地计算、合法委托或不同披露粒度才能完成。**知识状态改变未来行动约束**是本研究的序列性来源。

这一动机不要求所有敏感关联都禁止披露。例如同一关联若被明确授权用于治疗或特定业务用途，CI 评价应允许它；信息内容、主体、接收者、用途与传输原则须联合判断。已经知道某事实也不使再次违规发送自动合规。

### 2.2 直接相关方法与可借鉴机制

| 研究 | 已有方法要点 | 本方案借鉴与需要比较的部分 |
|---|---|---|
| CI-RL [R1] | 必要信息保留与禁止信息抑制的奖励，GRPO 训练 | 即时任务充分性与合规奖励；基线为同数据下的逐步 CI 训练 |
| PrivAct [R2] | 多 Agent 候选行动树、终局泄露/帮助性评分、价值传播、DPO | 模型内化训练与合规优先偏好；对照其训练信号如何反映历史组合 |
| CDI [R3] | 利用执行上下文的动态指导、失败状态截断、基于后续行动的 RL | 状态条件化反馈和 Appropriate Disclosure；比较外部指导与更新执行模型 |
| SelfCI [R4] | allowed/disallowed 互补教师与加权 KL 蒸馏 | 训练时同时推动正确保留和正确抑制；比较其在累计历史环境的行为 |
| Normative Simulacra [R5] | 规范判断、规范接地和对比环境奖励；处理 GRPO 奖励并列与弃答捷径 | 规范敏感奖励与反捷径质量控制 |
| ConCI [R6] | 受控反事实 Flip/Hold 三元组 | 控制相关与无关历史变化；不固定 H0/H1/H2 为三档奖励 |
| TWAIN [R7] | 双世界与轨迹级观察者推断奖励 | 长轨迹的信息后果；需要加入具体接收者、用途与允许信息流 |
| TG-CI [R8] | 接收者披露历史、任务图及累计/传递披露控制 | 历史记录与传播约束的强系统基线；区分控制与策略学习 |
| PrivacyFlow [R9] | 接收者背景知识、推断模拟、CI 分析及运行时规划 | 组合推断证据与任务完整性；作为推理时语义规划强对照 |
| PrivacyAlign / CIDER [R10,R11] | 人类隐私/遗漏偏好、披露粒度与可识别性 | 偏好质量、规范歧义、有效披露与过度拒绝控制 |

**论文可能的贡献中心：** 让执行模型学习 *recipient-state-conditioned long-horizon CI policy*，并从“即时合规”扩展至“当前合法行动对未来合法任务完成能力的影响”。CCE、DPR、反事实续行与信用分配是服务于此学习目标的算法机制。研究完成前，方法优越性及相对上述工作的新增效果均为待验证事项。

## 3. 正式问题设定

### 3.1 状态、观察、行动

系统包含多个角色 Agent \(i\in\{1,\ldots,n\}\)，可共享同一个语言模型 \(\pi_\theta\) 与 LoRA 参数，但拥有不同工具、私有数据与合法观察。定义完整环境状态：

\[
s_t=(q_t,N,D_t,Z_t,M_t),\qquad o_t^i=\operatorname{Observe}_i(s_t).
\]

其中 \(q_t\) 为当前任务和剩余子目标，\(N\) 为固定 CI 规范，\(D_t(r)\) 为接收者 \(r\) 实际收到的消息记录，\(Z_t\) 为任务进度、期限与资源状态，\(M_t\) 为分布式私有业务数据和工具状态。执行 Agent 根据角色和合法观察生成 \(a_t^i\sim\pi_\theta(\cdot\mid o_t^i)\)。

只有成功交付动作 \(\operatorname{Send}(i,r,m)\) 才更新对应接收者记录；草稿、计划和模拟候选均不能污染真实执行历史。评价器可读取完整模拟世界用于真值计算，但不能把隐藏真值注入执行模型输入。

### 3.2 接收者知识与 CI 违规

给定被明确建模的背景知识 \(B_r\)，接收者可推断知识为：

\[
K_t(r)=\operatorname{Closure}(B_r\cup\operatorname{Semantics}(D_t(r))).
\]

消息交付后产生新增推断（记为 \(\Delta K_t(r)\)，即新增知识集合与原知识集合的差）。按规范 \(N\) 分别计算：

- \(V_t^{\mathrm{direct}}\)：当前传输（发送者、接收者、主体、属性、用途、传输条件）本身违规；
- \(V_t^{\mathrm{infer}}\)：本步新增关系中存在该接收者在该用途下未被授权获得的事实。

\[
V_t=V_t^{\mathrm{direct}}\lor V_t^{\mathrm{infer}}.
\]

标签需同时包含直接与衍生事实的来源证据。推断证据缺失、语义冲突和规范模糊返回 `UNKNOWN`，不自动当作合规或违规。知识闭包在首版仅覆盖可检查的实体链接、关系组合与确定性蕴含；开放世界概率推断是扩展方向。

### 3.3 当前合法行动与长期可行性

定义 \(\mathcal A_{\mathrm{CI}}(s)\) 为当前可执行且符合 CI 的行动集合。动作空间除发送消息外，还允许合法概括、本地计算、请求任务条件、授权委托与必要的暂缓。某行动是否有价值，取决于它的后续任务路径与成本，而不能以行动集合大小直接代替。

对于给定期限 \(T\)，定义 \(\mathcal T_{\mathrm{safe}}(s)\) 为从状态 \(s\) 出发、后续所有交付均合规且最终满足任务条件的轨迹集合。\(\mathcal T_{\mathrm{safe}}(s)\neq\varnothing\) 只证明存在合规完成路径。进一步定义：

\[
Q_{\mathrm{safe}}^\star(s,a)=\max_{\pi'\in\Pi_{\mathrm{CI}}} \mathbb E_{\pi'}[U_T-\eta C_{t:T}\mid s,a],
\]

\[
Q^\pi(s,a)=\mathbb E_{\pi_\theta}[u(J(\tau))\mid s,a].
\]

\(Q_{\mathrm{safe}}^\star\) 是理想的合规最优后续价值；\(Q^\pi\) 描述当前团队模型实际执行的长期结果。开放式自然语言 Agent 环境通常难以精确求出 \(Q_{\mathrm{safe}}^\star\)，因此算法使用**固定参考续行策略** \(\pi_C\) 的估计 \(\widehat Q^{\pi_C}\) 和经过验证的合法成功路径见证；明确分开理论最优、固定策略估计与找到一条可行路径这三种证据。

## 4. 训练结果评价与 Reward–Data 合同

### 4.1 统一的行动/轨迹评价原则

每条轨迹保存三元结果：

\[
J(\tau)=(\operatorname{CICompliant}(\tau),\operatorname{TaskSuccess}(\tau),-\operatorname{ExecutionCost}(\tau)).
\]

默认优先级为：**全轨迹 CI 合规 → 任务成功与必要信息保留 → 合规成功路径中的实际成本**。对于随机执行，需保留违规概率、合规任务成功率、成本分布及不确定性，不能只报告一个奖励均值。成本只在具有可比任务结果的合规轨迹之间排序；无条件拒绝不能获得与合规完成任务相同的效用。

v0.7 将固定续行策略的价值明确写为

\[
Q_{\mathrm{CI}}^{\pi}(s,a)=\bigl(-p_{\mathrm{viol}}^{\pi},\ p_{\mathrm{safe\ success}}^{\pi},\ -\mathbb E_{\pi}[C\mid\mathrm{safe\ success}]\bigr).
\]

先比较风险，再比较合规成功，最后比较成功成本。每维保留样本量与不确定性；未解决的风险方向不能被成功/成本覆盖。未观察到违规不等于零风险，也不能把高优先级差异“不显著”当作等价。成本维在没有事前冻结的等价/非劣条件时只作描述；当前 discovery 入口不据此导出成本偏好。Qπ、精确有界 Q* 和成功见证严格分开。

即时检查、任务完成与长程评价使用**同一套 CI 真值与任务 oracle**，具体场景改变的是世界事实和任务规则，不能为制造偏好反转而手动给某类历史加分。

### 4.2 三类可学习监督

| 数据子集 | 生成条件 | 主要作用 |
|---|---|---|
| \(\mathcal D_{\mathrm{CI}}\) | 同一状态与合法观察下，候选存在明确的当前合规性差异，且优选行动有可验证任务充分性 | 当前知识条件化披露 |
| \(\mathcal D_{\mathrm{future}}\) | 两个当前均合规的行动，经**同前缀**参考续行得到可靠的合规任务成功/成本差 | 当前披露对长期合法路径的影响 |
| \(\mathcal D_{\mathrm{repair}}\) | 真实失败轨迹中找到同一决策状态下可执行、可验证更优的下游替代动作 | 纠正下游策略错误 |

H0/H1/H2 目前仅作为受控状态比较方式，不作为固定训练类别。额外保留**合法历史组合正例**与高度相似但无关的历史，以分别检查“见关联就拒绝”与“历史越长越保守”两类捷径。对所有组合，任务要求及合规偏好应有独立依据。

### 4.3 偏好来源的证据等级

- `exact_within_horizon`：在明示的有限动作、转移和期限内完成穷举，能比较该有界世界中的最优值/最小成本。
- `proven_unreachable_within_horizon`：相同有限模型内穷举证明期限内无合规路径；不得写成现实世界的普遍不可达性。
- `verified_witness_only`：找到完整合法成功续行，只证明存在；不能凭两条见证成本直接推断全局优劣。
- `paired_statistical_estimate`：冻结参考策略、相同外部随机种子、配对重复续行，价值差通过预先定义的置信条件；仅支持该固定参考策略下的偏好。
- `unknown`：搜索未覆盖、语义不确定、规则冲突或缺少任务 oracle；不导出严格偏好。

## 5. KCI-Align 方法框架

### 5.1 候选行动生成

生成器接口为 `generate(observation, limit) -> CandidateBatch`，只接收当前 Agent 的合法观察。模型提出实际语义差异明确的行动，包括完整披露、选择性披露、本地处理、合法委托、请求条件、适当暂停。按接收者、披露内容、动作类型及执行后状态去重；模型自报 `fact_ids` 或隐私自评不得直接作为事实标签。自由文本消息必须经独立语义提取与蕴含检查，不能只检查 JSON 参数。

生成器同时记录版本、采样配置、是否穷举和解析失败数。真实 LLM 生成器均视为非穷举；没有找到合规候选时返回“候选覆盖不足”，不能自动断言不存在合法行动。

### 5.2 算子一：CI-Conditioned Continuation Evaluation（CCE）

**输入：** 完整可重放状态 \(s_t\)、当前角色 \(i\)、候选行动 \(a\)、冻结的当前/参考策略版本、生成与搜索预算。  
**输出：** 即时审计结果、成功交付后的状态、当前策略续行价值、参考策略续行价值、合法路径见证、评分证据与不确定性。

核心步骤：

1. 验证该行动在角色当前观察和工具权限下可执行；在隔离沙盒中真正模拟交付；
2. 根据交付前后的接收者知识状态与 CI 规范检查直接/增量违规；遇到 `UNKNOWN` 停止生成硬标签；
3. 对当前合法动作，从更新后的状态由当前共享模型 \(\pi_{\theta_k}\) 继续执行，并用冻结参考策略 \(\pi_C\) 搜索合法续行；
4. 逐步验证后续每个 Agent 的观察权限、消息实际语义、交付结果、任务终局及成本；
5. 在**同一前缀、同一当前 Agent** 的两个动作之间，只在证据足够时导出长期偏好。

`CCE(s,i,a) -> {ci_verdict, knowledge_delta, Q_actual, Q_reference, witness, evidence_status, uncertainty, trace}`。

**有限计算原则：** 对明确即时违规、无效动作及未知语义不进行昂贵的合规参考搜索。状态池从真实 TRAIN 访问按事前规则选取，不按 CCE 胜负筛选。使用固定预算或事前冻结且控制多次查看错误率的增采样规则；不得看完结果后放宽门槛。缓存绑定世界状态、策略版本与搜索预算。

**新增监督入口：** `kci_cce_discovery_run.py analyze` 调用 `kci_unique_supervision.py`，逐状态保存 CE 已有动作、终局均值、同标量置信对照、CCE 的发现/验证结果及缺失状态。分别报告新观察、相对 CE 的覆盖、相对终局的方向差、跨块确认和每条有效偏好的完整费用。现有成功路径的标签冲突不能自动算教师错误；CE 在该观察无标签也不意味着它无法取得标签。

本轮可执行的 UUS 路径要求：记录的 CE 未覆盖、CCE 跨两个固定策略及独立采样块确认、终局方向未稳定重复同一动作、双方条件能力通过且费用完整。它只开放另行冻结的比较协议，不自动训练。改变标量目标为 CI-first 所造成的差异单独标注，下一次比较必须保留使用相同分叉的 CI-first 点估计教师，以隔离目标改变与置信筛选。预算效率或对已有 CE 的可靠纠错是另外两条假设，尚不能由本入口认定成立。

### 5.3 算子二：Downstream Policy Repair（DPR）

**输入：** 真实失败轨迹及对应 CCE 审计。  
**输出：** 失败后有可验证改善的**同状态同角色**行动偏好，或 `unresolved`。

流程为：定位首次已发生的 CI 违规或已确认的任务失败；沿实际前缀从相关决策处回溯；从当时 Agent 的合法观察提出替代行动；使用 CCE 复核可执行性、CI 与后续任务充分性；在证据支持下生成 `(prompt, chosen, rejected, metadata)`。如果 B 能在已有接收者历史下改为本地处理并合法完成任务，就训练 B 在当时的决策；A 先前合法的信息流可另由 CCE 的长期路径比较形成独立偏好。DPR 不把参考轨迹与实际轨迹的第一个文字差异自动当成错误，因为多条行动路径可能同样正确。

### 5.4 训练优化器

v0.7 先使用已有验证成功轨迹建立共享 Action SFT 执行器，验收必要保护、合法 RAW、授权例外和完整流程。监督来源实验固定相同优化器，以区分监督获取与优化器效果；DPO 保留为后续可比较的更新方式，不预设其优于 CE。三类监督保留独立来源与消融，两个候选必须拥有同一可见 Prompt、同一状态、同一角色。\(\pi_0\) 为 DPO 使用时的冻结基准模型，\(\pi_C\) 为参考续行策略，二者不能混用。

\[
\mathcal L_{\mathrm{KCI}}=\lambda_1\mathcal L_{\mathrm{CI}}+\lambda_2\mathcal L_{\mathrm{future}}+\lambda_3\mathcal L_{\mathrm{repair}}.
\]

上式仅表示监督来源的分解，不表示三类损失已同时得到实证支持。当前不叠加新的历史联合损失；DPR 暂不进入新训练，只保留真实失败位置、合法替代和完整后果验证的接口。任何 CCE-DPO 更新都须先通过新增监督与参考能力门，再冻结相同初始模型、状态池、预算及强基线。不能通过任意标量化抵消风险优先级。

## 6. 数据来源、标注与环境设计

### 6.1 现有数据的实际复用边界

| 数据来源 | 已核查字段与优势 | 本研究必须补建的部分 |
|---|---|---|
| PiSAs [R12] | `scenario.json` 中角色与预置资料，`utility.json` 中任务 oracle，`appropriateness.json` 与 `visibility.json` 中属性级规范；Meeting/Jira 有敏感原文与可用业务约束的配对 | 实际 A2A 交付日志、派生关系的 CI 授权、状态分叉、行动级效用与参考续行 |
| MultiAgentPrivacy [R13] | 多 Agent 分布的表和实体键、组合推断步骤、正常/敏感目标 | CI 五元关系与授权真值、合法替代行动、动态交付日志、可验证任务奖励 |
| ConCI / CIDER [R6,R11] | 受控条件变化、披露粒度与用户偏好 | 真实多步执行、知识更新和长期任务结果；可用于外部诊断 |

特别注意：PiSAs 原始 `timeline` 是预置组织资料，**不是执行时成功交付的消息历史**。当前 PiSAs 对原子属性的 `appropriate` 标签无法自动推导“相同消息在不同历史下是否导致未授权的新关系”。例如真实 Meeting 样本的 `a9`（医疗原因）与 `a10`（房间约束），Jira 样本的 `a22`（绩效情况）与 `a22n`（受限业务约束），可支持选择性披露监督，但需要新增组合关系与动态规范标签才能用于 KCI 的长期训练。

### 6.2 最小数据合同

每条 episode 至少保留以下字段：

`world_id, rule_family_id, episode_id, task, purpose, role_acl, norm_version, source_facts, task_oracle, delivered_events, current_state_id, acting_agent, authorized_observation, candidate_action, actual_message_semantics, derived_facts, ci_audit, task_outcome, continuation_evidence, reference_policy_version, candidate_policy_version, search_budget, preference_evidence_status`。

构造标签时严格保证：先前实际交付合法；组合推断关系可复核；派生事实是否获授权有明确规范；成功/部分成功/失败由任务 oracle 判定；替代行动在当前角色可执行；引用的参考续行有完整轨迹证据。对于社会规范歧义，单独标注并人工仲裁；无法确定的案例不作为硬奖励。样本划分按基础世界、关系规则家族以及任务模板分组，相关事实换名、顺序变化、语义改写不得跨训练/测试集泄漏。

### 6.3 跨任务覆盖

训练/评价应覆盖多个符合以上合同的协作任务，至少包含信息汇集、资源协调、内部消息转交与必要的本地计算。第一版结构化任务保证可验证性；后续加入自然语言变体、复杂工具调用和不同任务模板，用于检验方法是否只依赖特定标识符或业务措辞。项目分配或预约案例可放在论文 Introduction 解释累计知识与后续路径的变化，但不限制问题定义。

## 7. 实验问题、评价指标与基线计划

### 7.1 研究问题

- **RQ1：** 在相关与无关历史变化下，模型是否能够根据接收者知识正确改变/保持实际披露行动？
- **RQ2：** 长期合规续行监督是否提高全轨迹合规任务完成率，优于仅使用即时 CI 合规与任务充分性奖励？
- **RQ3：** DPR 是否帮助纠正下游执行错误，并减少早期合法行动被错误压低概率的情况？
- **RQ4：** 学到的策略能否迁移到新的关系模板、消息顺序、角色组合、任务领域及较长交互？

### 7.2 主要指标

| 指标 | 含义 | 最重要的解释 |
|---|---|---|
| Safe Task Success Rate | 全轨迹 CI 合规且任务正确完成的比例 | 核心主指标，拒绝无法获得高分 |
| Direct / Incremental CI Violation | 直接违规、组合后新增未授权推断发生率 | 区分失败来源 |
| Task Success / Required-Information Coverage | 任务结果与必要信息保留 | 单独评估任务损失 |
| Legal Completion Cost | 合规成功轨迹的消息、工具、期限成本 | 比较路径效率 |
| History-Conditioned Action Accuracy | 相同世界、不同接收者历史下行动是否与规范和任务一致 | 评价条件敏感性，不强制文本一致 |
| Shortcut / Over-refusal Rate | 无关历史、合法组合、长度和敏感词扰动下的不必要拒绝 | 识别奖励捷径 |
| Reference Evidence Coverage | 有多少状态能产生精确/统计/见证/未知证据 | 检查实际监督有效规模与偏差 |

### 7.3 必要比较与消融

在相同基础世界、任务、模型、训练量、历史可见性和验证器下比较：原始/提示模型、轨迹 SFT、逐步 CI 奖励训练（CI-RL 或适配基线）、普通任务条件化 DPO、PrivAct 式多 Agent 偏好基线、全轨迹 CI×任务奖励、KCI-Align 的仅 \(\mathcal D_{\mathrm{CI}}\)、加 \(\mathcal D_{\mathrm{future}}\)、再加 \(\mathcal D_{\mathrm{repair}}\)。TG-CI / PrivacyFlow 作为运行时系统强对照，明确记录额外规划/验证开销和是否在执行阶段使用外部隐私中介。

特定验证器或数据未开放时如实标注适配程度，不能虚构与原论文完全复现的数值。比较应报告样本成本、模型调用数、训练计算量、多种随机种子和统计区间；在可精确求解的子集上分析 \(\widehat Q^{\pi_C}\) 与最优有界值的偏差。

## 8. 现有实现状态

### 8.1 v0.7 的真实证据与缺失

- 共享 4B 全流程 SFT 后，同十四个上游状态和旧 K8 规则的偏好接受由 0/14 变为 7/14；七条与 CE、终局方向一致，不构成 CCE 专属收益。
- 同标签三臂保留为 legacy 负对照；旧训练入口默认退出，仅显式历史复现可运行。即时 release 的静态可见闭包是必须保留的强基线。
- 基础曝光修复候选条件续行完成 65/73→70/73，但合法 RAW 5/7→3/7，未过能力门，未替换为稳定参考策略。该面板来自 TRAIN，不是未见任务证据。
- v0.7 开发加入监督准入、真实轨迹重放及空标签阻断；这些是流程修复，不是新方法性能结果。旧发现实验的参考策略未完成，不能补写为已验证。
- KC8 的独立根来源与 PiSAs 派生规范仍需独立核验。DG36、P211/P212 等封存根不进入开发；反复使用过的 DEV 不再称为新的泛化证据。
- **Main Results / Ablation 仍缺失**：CCE 相对强 SFT/终局/可见规划器的独立增益、DPR 消融、外部自然规范与独立任务泛化均未建立。

### 8.2 Legacy：2026-11-12 接口与玩具环境

已经生成并在本地复核：`KCI-Align_v0.5_Implementation.zip`、`KCI_Align_v0_5/SPEC_v0.5.md` 与 `README.md`。当前 Python 包包含 `kci/candidates.py`、`continuation.py`、`paired.py`、`repair.py`、`rewards.py`、`datasets.py`、`toy_env.py` 等接口。

**本次重新运行结果：** `python -m unittest discover -s tests -v` 共 **24/24 项通过**。`examples/run_demo.py` 在人为构造的有限环境中复现了两条合规路径的最小动作成本分别为 **4 与 7**，以及下游 B 的组合泄露可被 DPR 在 B 的决策状态修正；演示导出 **2 条 DPO 偏好**。这些是玩具环境的程序正确性检查。

**尚未完成：** 真实 LLM 推理接入、自由文本语义标注与人工校准、跨任务动态数据构建、真实参考续行成本统计、共享 LoRA/DPO 训练、完整基线对比和独立泛化评价。

## 9. 技术风险、证据标准与阶段里程碑

| 风险 | 具体来源 | 控制方式与项目决策点 |
|---|---|---|
| 历史状态与标签不匹配 | 把预置资料当作已交付消息；推断关系未获规范确认 | 强制事件重放、规范版本与证据链检查；标签不明为 UNKNOWN |
| 历史依赖监督太弱 | 安全保守路径在所有状态都同样完成任务 | 保留真实任务价值差异；合法组合正例；不人为赋分制造 flip |
| 参考续行偏差 | 搜索失败被误判不可行；教师越权看见 oracle | 区分穷举、见证、统计和未知；每步验证权限与可执行性 |
| 训练信号退化 | 当前动作奖励无差异、GRPO 组内并列、拒绝捷径 | 奖励分布审计、任务充分性、难例与候选覆盖率 |
| 算法成本过高 | 每个候选需要多次长轨迹续行 | 有界搜索、缓存、仅比较关键候选、按不确定性增采样 |
| 真实任务迁移差 | 训练只学会有限的实体连接/固定模板 | 任务/规则家族外测试、语义改写、不同顺序/长度/模型 |
| 论文贡献不突出 | 普通 DPO 在同等数据和历史下已达到相同表现 | 直接报告；重点检验长期可行性与 DPR 的独立增量 |

推荐里程碑：**M1** 完成通用数据与规范合同；**M2** 构建有可验证真实历史依赖的多任务环境；**M3** 完成参考续行校准及偏好证据质量审计；**M4** 打通共享模型 SFT/DPO 与即时 CI 基线；**M5** 测试长期价值与 DPR 的独立增量；**M6** 扩展自由文本与外部任务，并整理论文证据。每个里程碑的通过条件以接口验证、标签质量与完整实验记录为准，不以完成代码行数替代研究结果。

## 10. v0.7 实现交接与停止条件

旧 v0.5 接口包仅作历史参考。当前按以下依赖推进：

1. **可靠共享执行器：** 复用已有成功轨迹与业务根，补足必要保护和合法 RAW 的条件对照。不得以重复采样稀有动作替代独立监督；当前曝光修复候选未通过，不作为 CCE 稳定参考。已淘汰的联合历史损失不再调参。
2. **长期监督准入：** 冻结 TRAIN 访问、当前合法分叉、两个续行策略和费用预算；运行可执行 UUS 入口及真实重放。保留完全等价、普通成本差、未来风险/成功差与 UNKNOWN 分类。KC8 与 PiSAs 仅在来源、观察、交付、规范、成功见证和成本对齐后接入，不盲目重生成。
3. **一次有裁决力的比较：** 仅在获得非冗余可靠信号后，冻结同底座、状态、候选和总费用的强 Action SFT、即时 CI、终局标量、CI-first 点估计及 CCE 对照；记录风险、合规完成、合法 RAW、过度核验、Flip/Hold、角色与成本。DPR 必须另有新状态/新信息才进入消融。无增量则停止当前 CCE-DPO 创新主张，不能再重复同标签模型训练。

**Codex 交付准则：** 每阶段提交修改文件、接口说明、自动测试、可复现实例、未解决问题与下一阶段风险；每次新增数据字段要指出其用于哪条奖励/偏好监督。禁止把搜索不到路径标为负例、把模型生成的 `fact_ids` 当真值、把已知授权规范写成所有模型都可见的隐藏 oracle，也不要将玩具环境成功报告为实证创新。

## 11. 预期贡献与最终论文叙事

研究完成后，拟围绕三项可测量贡献组织论文：

1. **动态 CI 学习问题与形式化：** 以接收者知识状态转移解释多步披露的即时规范适当性与长期合规任务价值，提供与任务完成结果共同定义的评价单位。
2. **KCI-Align 训练方法：** 由 CCE 产生状态条件化即时/长程行动监督，利用 DPR 从真实失败续行中提取可信修正，训练共享参数执行模型内化长期 CI 策略。
3. **可验证的多任务动态监督和证据：** 在多 Agent 信息分布、合法历史、语义组合与任务结果之间建立可追溯标签与评价，检验跨任务/历史/角色泛化及对过度拒绝的影响。

**论文成败标准：** 在严格匹配的数据与模型条件下，相比逐步 CI 奖励和普通长轨迹奖励，长期监督需要带来清晰、稳健且可解释的改进。若差异不成立，优先回到奖励—数据匹配、可观测状态、参考估计质量和序列任务约束进行机制审查。

## 参考文献与可追溯来源

以下为本提案主要使用的论文及公开实现；编号用于文中定位。用户提供的 28 篇论文 PDF 是正文与机制分析的主要依据，公开仓库用于代码与字段核对。部分论文没有在本轮重复检索其正式出版元数据，引用时以正文标题为准。

- **[R1]** Lan et al. *Contextual Integrity in LLMs via Reasoning and Reinforcement Learning.* arXiv:2506.04245. https://arxiv.org/abs/2506.04245
- **[R2]** Cheng et al. *PrivAct: Internalizing Contextual Privacy Preservation via Multi-Agent Preference Training.* https://arxiv.org/abs/2602.13840 ; https://github.com/chengyh23/PrivAct
- **[R3]** Wen et al. *Contextualized Privacy Defense for LLM Agents* (CDI). https://arxiv.org/abs/2603.02983 ; https://github.com/SALT-NLP/contextual_privacy_defense
- **[R4]** Park et al. *It Takes Two: Complementary Self-Distillation for Contextual Integrity in LLMs.* https://arxiv.org/abs/2605.20258 ; https://github.com/sw-programmer/SelfCI
- **[R5]** Franchi et al. *Context-based normative simulacra from narrative fiction.* COLM 2026. https://arxiv.org/abs/2604.20904
- **[R6]** Park et al. *ConCI: A Controlled Counterfactual Benchmark for Contextual Integrity in LLMs.* https://arxiv.org/abs/2610.24494
- **[R7]** *Indistinguishable Agents: Training Agent Privacy as Twin-World Indistinguishability* (TWAIN). https://arxiv.org/abs/2610.25161
- **[R8]** Wu et al. *Judging Before Propagation: Task-Graph Contextual Integrity for Multi-Agent Privacy.* https://arxiv.org/abs/2610.39532
- **[R9]** Staab et al. *PrivacyFlow: Privacy in AI Agents through Information Flow Analysis and Consequence-Aware Risk Management.* https://arxiv.org/abs/2610.40714
- **[R10]** *PrivacyAlign: Contextual Privacy Alignment for LLM Agents.* https://arxiv.org/abs/2606.21710
- **[R11]** Guo et al. *CIDER: A Dataset of Contextual Disclosure Boundaries for Privacy Preference Alignment.* https://arxiv.org/abs/2608.09164 ; https://huggingface.co/datasets/peach-lab/CIDER
- **[R12]** *PiSAs: Benchmarking Contextual Integrity in Multi-User Agentic Systems.* https://arxiv.org/abs/2607.05318 ; https://huggingface.co/datasets/ServiceNow/PiSAs ; https://github.com/ServiceNow/PiSAs
- **[R13]** *The Sum Leaks More Than Its Parts: Compositional Privacy Risks and Mitigations in Multi-Agent Collaboration.* Data and code: https://github.com/Vaidehi99/MultiAgentPrivacy
- **[R14]** Foerster et al. *Counterfactual Multi-Agent Policy Gradients* (COMA). https://arxiv.org/abs/1705.08926
- **[R15]** Arjona-Medina et al. *RUDDER: Return Decomposition for Delayed Rewards.* https://arxiv.org/abs/1806.07857

---

**本文档版本关系：** v0.7 在 v0.6 研究目标上收紧监督准入与证据边界，统合此前 Task Specification v0.1、Reward Specification v0.2/v0.3 及实现接口 v0.5。研究问题与算法合同不绑定特定业务任务。若多个预先封存独立根上的公平比较持续无非冗余监督或方法收益，停止当前 CCE-DPO 的算法创新主张；环境/评测贡献另行审查，不能未经验证改称完整方法成功。
