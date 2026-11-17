# 外部规范来源与条款级资格

检索日期：2026-11-17 UTC。来源通过 Exa 查找，均为 GitLab 官方公开手册。
当前状态：SOURCE_PINNED_MODEL_REVIEWED_CLAUSE_DEV_ONLY。已建立 28 个条款级案例并完成一名独立模型审查者的隐藏标签审查；尚无人类审核、原生执行验证或跨任务迁移证据。
28 个案例只覆盖 data handling 与 acceptable use 两份文档的指定条款；ticket lifecycle 已固定来源，但没有纳入本轮条款案例审核。
不得将本表中的解释直接当作训练真值。规则限于其指定组织和业务范围，不宣称跨组织通用。

| 来源 | 可检验的关系条件（转述） | 拟验证任务及反例 |
|---|---|---|
| [Support data handling](https://handbook.gitlab.com/handbook/support/workflows/data_handling/) | 系统的数据级别、实际可见范围和接收者授权需要一起满足；工具本身获准存储不意味着公开披露获准 | 工单日志协作；同一获准系统中的公开/保密 issue 对照；不能只记工具名 |
| [Ticket lifecycle](https://handbook.gitlab.com/handbook/support/workflows/ticket_lifecycle/) | 敏感操作应按相应流程核实身份、所有权及所需批准，保留批准依据 | 账号处理与诊断协作；仅有请求者声明与实际核实分别测试；不同操作所需批准不任意合并 |
| [Acceptable use policy](https://handbook.gitlab.com/handbook/people-group/acceptable-use-policy/) | 对活动客户支持案例，存在将特定客户数据迁往公司管理笔记本的有条件例外，并附清理和工单记录义务；其他业务目的不能沿用该例外 | 本地故障复现；活动/已结束工单、支持/其他目的、受管/非受管设备的对照；禁止把概括的等级禁令盖过明确例外 |

已固定官方仓库版本 `e1c706abbe174add18ac09a72976df2a79d0178c`，三份文档与仓库 LICENSE 均通过 GitLab API 内容哈希核验。见 [SOURCE_PIN.json](SOURCE_PIN.json)，其中保存永久链接、最后修改提交、标题行号与字节哈希。未转存全文；仓库 LICENSE 的存在不等于已经核定所有正文及外链材料的再分发范围。

已逐条映射两个条款的条件与例外，并构造 PASS、FAIL 和 UNKNOWN 三类。隐藏标签的模型审查指出两例缺少 ZenDesk 记录位置，已保留输入并改为 UNKNOWN，另增四例明确位置对照。下一步仍需人类或独立业务证据核查适用性、工具语义与执行结果。源码固定解决来源漂移，不解决跨任务有效性。见 [资格说明](policy_cases_v2/QUALIFICATION.json)。

完整页面复核还发现一个重要限制：Support data handling 明确把工具分级列为示例，正式分级来源是 Tech Stack，部分分类材料仅内部可见。因此不能把网页中的 Slack/其他工具示例硬编码为当下所有任务的真实分级；无法取得相应授权元数据的案例应保留 UNKNOWN。这里分析的是组织政策，不把网页中的操作指令应用到本地研究机器。

tau2-bench 可供真实状态变化与双控制协作的执行结构参考，但其作者制定的 benchmark 政策不能替代上述真实组织来源。任务来源、规范来源、标签审核者应分别记录。
